import express from "express";
import bodyParser from "body-parser";
import fetch from "node-fetch";
import * as cheerio from "cheerio";
import OpenAI from "openai";
import admin from "firebase-admin";
import fs from "fs";
import { Pinecone } from "@pinecone-database/pinecone";

import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(bodyParser.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-large";
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "LumiChat";
const PINECONE_CLOUD = process.env.PINECONE_CLOUD || "aws";
const PINECONE_REGION = process.env.PINECONE_REGION || "us-east-1";
const EMBEDDING_DIMENSION = Number(process.env.EMBEDDING_DIMENSION) || (EMBEDDING_MODEL === "text-embedding-3-small" ? 1536 : 3072);

admin.initializeApp({
  credential: admin.credential.cert(JSON.parse(fs.readFileSync("./firebaseServiceAccount.json", "utf8")))
});
const firestore = admin.firestore();

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

async function ensurePineconeIndex() {
  const existing = await pinecone.listIndexes();
  const exists = existing.indexes?.some(i => i.name === PINECONE_INDEX_NAME);
  if (!exists) {
    console.log(`Creating Pinecone index ${PINECONE_INDEX_NAME}...`);
    await pinecone.createIndex({
      name: PINECONE_INDEX_NAME,
      dimension: EMBEDDING_DIMENSION,
      metric: "cosine",
      spec: {
        serverless: {
          cloud: PINECONE_CLOUD,
          region: PINECONE_REGION,
        },
      },
    });
    // Wait until the index is ready
    let ready = false;
    while (!ready) {
      const describe = await pinecone.describeIndex(PINECONE_INDEX_NAME);
      if (describe.status?.ready) ready = true;
      else await new Promise(r => setTimeout(r, 2000));
    }
    console.log(`Pinecone index ${PINECONE_INDEX_NAME} is ready.`);
  }
}

async function seedIfEmpty() {
  try {
    // Ensure index exists if allowed
    let exists = false;
    try {
      const existing = await pinecone.listIndexes();
      exists = existing.indexes?.some(i => i.name === PINECONE_INDEX_NAME);
    } catch (e) {
      console.warn("listIndexes failed", e?.message || e);
    }
    if (!exists) {
      if (process.env.ENSURE_INDEX_ON_STARTUP === "true") {
        await ensurePineconeIndex();
      } else {
        console.warn(`Index ${PINECONE_INDEX_NAME} not found and ENSURE_INDEX_ON_STARTUP is not true; skipping auto seed.`);
        return;
      }
    }

    const index = pinecone.index(PINECONE_INDEX_NAME);
    const stats = await index.describeIndexStats({});
    const count = stats?.totalRecordCount || 0;
    if (count === 0) {
      console.log("Pinecone index is empty; auto seeding now...");
      await scrapeAndIndexDocs();
      await indexPastTickets();
      console.log("Auto seed completed.");
    } else {
      console.log(`Pinecone index already has ${count} vectors; skipping auto seed.`);
    }
  } catch (e) {
    console.error("seedIfEmpty error", e);
  }
}

function chunkText(text, chunkSize = 800, overlap = 200) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    if (end === text.length) break;
    start += Math.max(1, chunkSize - overlap);
  }
  return chunks;
}

async function getEmbedding(text) {
  console.log(`Getting embedding`);
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text
  });
  return res.data[0].embedding;
}

async function getEmbeddings(texts) {
  console.log(`Getting batch embeddings: ${texts.length}`);
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts
  });
  return res.data.map(d => d.embedding);
}

function cosineSimilarity(a, b) {
  console.log(`Finding cosine similarity`);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function scrapeAndIndexDocs() {
  const urls = [
    "https://docs.illumibot.ai",
    "https://docs.illumibot.ai/whats-new",
    "https://docs.illumibot.ai/projector-selection",
    "https://docs.illumibot.ai/first-time-setup",
    "https://docs.illumibot.ai/account-creation",
    "https://docs.illumibot.ai/home/",
    "https://docs.illumibot.ai/point/",
    "https://docs.illumibot.ai/play/",
    "https://docs.illumibot.ai/contests/",
    "https://docs.illumibot.ai/user-profile/",
    "https://docs.illumibot.ai/consumables/",
    "https://illumibot.ai/shop",
  ];
  console.log("Starting documentation indexing...");
  let indexedCount = 0;

  const processUrl = async (url) => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    const text = $("body").text().replace(/\s+/g, " ").trim();

    const chunks = chunkText(text, 800, 200);
    // Upsert in batches to Pinecone (deterministic IDs overwrite)
    const index = pinecone.index(PINECONE_INDEX_NAME);
    const batchSize = 100;
    let localCount = 0;
    for (let start = 0; start < chunks.length; start += batchSize) {
      const end = Math.min(start + batchSize, chunks.length);
      const batchChunks = chunks.slice(start, end);
      const embeddings = await getEmbeddings(batchChunks);
      const vectors = embeddings.map((embedding, j) => {
        const i = start + j;
        const chunk = batchChunks[j];
        const docId = `${encodeURIComponent(url)}_${i}`;
        return {
          id: docId,
          values: embedding,
          metadata: {
            type: "doc",
            source: url,
            text: chunk,
          },
        };
      });
      await index.upsert(vectors);
      localCount += batchChunks.length;
    }
    console.log(`Indexed docs from ${url}`);
    return localCount;
  };

  try {
    const concurrency = Number(process.env.SCRAPE_CONCURRENCY) || 3;
    for (let i = 0; i < urls.length; i += concurrency) {
      const slice = urls.slice(i, i + concurrency);
      const results = await Promise.allSettled(slice.map(url => processUrl(url)));
      for (const r of results) {
        if (r.status === "fulfilled") indexedCount += r.value;
        else console.warn("Scrape failed:", r.reason?.message || r.reason);
      }
    }
    console.log(`Documentation indexing completed. Indexed ${indexedCount} chunks.`);
    return { success: true, indexedCount };
  } catch (error) {
    console.error("Error in scrapeAndIndexDocs:", error);
    throw error;
  }
}

async function indexPastTickets() {
  console.log("Starting ticket indexing...");
  const snapshot = await firestore.collection("tickets").get();
  let indexedCount = 0;

  const items = [];
  for (const doc of snapshot.docs) {
    const data = doc.data();
    if (!data.messages || data.messages.length === 0) continue;

    const customerId = data.messages[0].created_by;
    const customerMessages = data.messages.filter(m => m.created_by === customerId);
    const systemMessages = data.messages.filter(m => m.created_by !== customerId);

    if (customerMessages.length && systemMessages.length) {
      const customerMsg = customerMessages[0].message;
      const systemReply = systemMessages[0].message;
      if (customerMsg && systemReply) {
        const combined = `Q: ${customerMsg}\nA: ${systemReply}`;
        items.push({
          id: `ticket_${doc.id}_0`,
          text: combined,
          metadata: {
            type: "ticket",
            ticket_id: doc.id,
            ticket_text: customerMsg,
            reply_text: systemReply,
            combined_text: combined,
          },
        });
      }
    }
  }

  const index = pinecone.index(PINECONE_INDEX_NAME);
  const batchSize = 100;
  for (let start = 0; start < items.length; start += batchSize) {
    const end = Math.min(start + batchSize, items.length);
    const batch = items.slice(start, end);
    const embeddings = await getEmbeddings(batch.map(b => b.text));
    const vectors = embeddings.map((embedding, i) => ({
      id: batch[i].id,
      values: embedding,
      metadata: batch[i].metadata,
    }));
    await index.upsert(vectors);
    indexedCount += batch.length;
  }
  console.log(`Ticket indexing completed. Indexed ${indexedCount} tickets.`);
}

async function searchSimilar(queryEmbedding, type, topK = 5) {
  const index = pinecone.index(PINECONE_INDEX_NAME);
  const result = await index.query({
    vector: queryEmbedding,
    topK,
    includeMetadata: true,
    filter: { type },
  });
  return result.matches || [];
}

app.post("/answer", async (req, res) => {
  try {
    const { ticketText } = req.body;
    const queryEmbedding = await getEmbedding(ticketText);

    const docMatches = await searchSimilar(queryEmbedding, "doc", 3);
    const ticketMatches = await searchSimilar(queryEmbedding, "ticket", 3);
    console.log(`/answer retrieval sizes → docs: ${docMatches.length}, tickets: ${ticketMatches.length}`);
    const context = [...docMatches, ...ticketMatches]
      .map(m => (m.metadata && (m.metadata.text || m.metadata.combined_text)) || "")
      .join("\n---\n");

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are Illumibot's tech support assistant. Use provided context to answer accurately and in human way." },
        { role: "system", content: `Context:\n${context}` },
        { role: "user", content: ticketText }
      ]
    });

    res.json({ answer: completion.choices[0].message.content });
  } catch (e) {
    console.error("/answer error", e);
    res.status(500).json({ error: e?.message || "Internal Server Error" });
  }
});

app.post("/reindex", async (req, res) => {
  try {
    if (process.env.ENSURE_INDEX_ON_STARTUP !== "true") {
      await ensurePineconeIndex();
    }
    await scrapeAndIndexDocs();
    await indexPastTickets();
    res.json({ ok: true });
  } catch (e) {
    console.error("/reindex error", e);
    res.status(500).json({ error: e?.message || "Internal Server Error" });
  }
});

app.get("/debug/index-stats", async (req, res) => {
  try {
    const index = pinecone.index(PINECONE_INDEX_NAME);
    const stats = await index.describeIndexStats({});
    res.json(stats);
  } catch (e) {
    console.error("/debug/index-stats error", e);
    res.status(500).json({ error: e?.message || "Internal Server Error" });
  }
});

(async () => {
  if (process.env.ENSURE_INDEX_ON_STARTUP === "true") {
    await ensurePineconeIndex();
  }
  if (process.env.REINDEX === "true") {
    console.log("Reindexing docs and past tickets...");
    await scrapeAndIndexDocs();
    await indexPastTickets();
  } else if (process.env.AUTO_REINDEX_ON_EMPTY !== "false") {
    // Non-blocking auto seed when index is empty
    seedIfEmpty();
  }
})();

app.listen(process.env.PORT || 3000, () => {
  console.log(`Server running on port ${process.env.PORT || 3000}`);
});

