# Architecture — plain-English overview

This document explains how the Oscar RAG app is put together, without
jargon. If you want the technical spec, read [`README.md`](README.md).
If you want to set it up on a fresh laptop, read [`tutorial.md`](tutorial.md).

---

## What the app does

You type a question about the 2026 Academy Awards. You choose **Graph
RAG** or **Text RAG**. You get an answer. Both modes use the same
language model but they look for information in very different places.

- **Graph RAG** answers questions about **who won what** — using a
  structured database of facts.
- **Text RAG** answers questions about **how the Academy works** — using
  a document with the institution's history.

The two data sources don't overlap on purpose. The graph knows nothing
about the Academy's voting process. The text knows nothing about who
won Best Picture. That's what makes the demo interesting — the same
question gives you very different answers depending on which mode you
pick.

---

## The three moving parts

```
 ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
 │   The user    │    │  The engine   │    │  The memory   │
 │  (a browser)  │ ─▶ │  (Flask app)  │ ─▶ │  (2 stores)   │
 └───────────────┘    └───────────────┘    └───────────────┘
```

1. **The browser** — a simple chat page. You type, you pick a mode, you
   see an answer.
2. **The Flask app** — one Python program running on your laptop. It
   decides which pipeline to run and calls the language model.
3. **The memory** — two databases living on your laptop:
    - **GraphDB** holds structured facts as an RDF graph (winners, films,
      nominations).
    - **Weaviate** holds the text corpus, chopped into paragraphs, each
      turned into a mathematical vector so we can search by *meaning*.

The language model itself (**Qwen 2.5 Coder 7B**) runs inside a
program called **Ollama**, also on your laptop. Nothing ever leaves
the machine.

---

## Graph RAG, step by step

You type: *"Which film won Best Picture?"*

1. **The model reads the graph's dictionary.** At startup, we hand the
   language model the ontology — a description of all the categories
   and relationships in the graph (things like "film", "director",
   "wonAward"). Think of it as the model's map of what's in the
   database.
2. **The model writes a database query.** Given your question and the
   dictionary, the model writes a SPARQL query — SPARQL is the query
   language for RDF graphs, like SQL is for spreadsheets. Something
   like *"give me the film that won Best Picture."*
3. **The database runs the query.** GraphDB looks up the answer:
   *"One Battle After Another."*
4. **The model turns the raw answer into a sentence.** The database
   returns something like `[('One Battle After Another',)]` — a
   Python-style raw result. The model rewrites it as *"One Battle
   After Another won Best Picture."*
5. **You see the answer, plus the SPARQL query the model wrote.**
   Clicking "Show generated SPARQL" reveals the query so you can
   verify the model wasn't making things up.

Two important details:
- The model is called **twice** — once to write the query, once to
  format the result. Different jobs, same model.
- Between those calls, the actual database does the work. The model
  never invents facts; it just translates your question into a
  precise query and reads the result.

---

## Text RAG, step by step

You type: *"How does the Academy voting process work?"*

Before you ever asked a question, we did some preparation once,
offline:

- We took the text file, chopped it into paragraphs of about 800
  characters each (roughly 60 chunks total).
- For each chunk we ran a tiny AI model called **MiniLM** that turns
  text into a list of 384 numbers — a *vector*. Two chunks about
  similar topics end up with similar vectors.
- We stored every chunk plus its vector in Weaviate.

Now when you ask a question:

1. **We turn your question into a vector** with the same MiniLM
   model. Your question now lives in the same "meaning space" as the
   chunks.
2. **Weaviate finds the nearest chunks.** It compares your question's
   vector to every chunk's vector and returns the closest 9. Closest
   in vector space usually means closest in meaning.
3. **A second, smarter model ranks them precisely.** The first search
   is fast but rough. A model called a **cross-encoder** reads each
   `(question, chunk)` pair together and gives it a precise
   relevance score. We keep the top 3 after this rerank.
4. **Safety check: how close is the closest chunk?** If even the
   nearest chunk isn't close enough to the question (measured by a
   distance number), we refuse to answer. We show a red warning and
   never call the language model. This is the single biggest
   protection against hallucination — if we didn't find anything
   relevant, saying "I don't know" is safer than making something up.
5. **The language model writes an answer using only those 3 chunks.**
   We give the model strict instructions: use only the passages we
   provided, cite the source with a marker like `[doc_5]`, and if
   the passages don't answer the question, say so honestly.
6. **We check the model's answer for citations.** If the answer
   contains at least one `[doc_N]` marker, we mark it as
   **"grounded"** (green). If it doesn't cite anything, we mark it as
   **"low confidence"** (yellow) — the model might have made things
   up despite our instructions.

You see one of three colored bubbles depending on how much we trust
the answer:

- 🟥 **Red — Abstained.** We didn't call the model at all. The
  question is outside what the corpus knows about.
- 🟨 **Yellow — Low confidence.** The model answered but didn't cite
  any sources. Be skeptical.
- 🟩 **Green — Grounded.** The model answered and cited at least one
  passage. Trustworthy.

---

## Why we split the data this way

If both pipelines had the same information, comparing them would be
useless. So we deliberately kept them separate:

- The **graph** contains *only* structured facts: dates, names,
  award categories, numbers. Perfect for questions like *"who won
  X?"* or *"how many nominations did Y get?"*
- The **text** contains *only* institutional history: how the
  Academy voting works, how the branches are structured, its
  origins. Perfect for questions like *"how does the Academy
  work?"* or *"what changed in recent years?"*

Ask *"Tell me about the 2026 ceremony"* to both pipelines and
you get two very different answers — one giving you dates and
winners, the other giving you context and significance. That
contrast is the whole point.

---

## What happens under the hood in one request

Let's trace *"What is AMPAS?"* in Text RAG mode from click to answer:

```
1. Browser sends POST /ask/text {"question": "What is AMPAS?"}
                       ↓
2. Flask receives the request. Checks its cache — has this exact
   question been asked before? If yes, returns the cached answer
   instantly. If no, continues.
                       ↓
3. Flask calls text_rag.ask("What is AMPAS?"):
                       ↓
4. MiniLM turns "What is AMPAS?" into a 384-number vector.
                       ↓
5. Weaviate finds the 9 chunks closest to that vector.
                       ↓
6. The cross-encoder reranks: it reads each (question, chunk) pair
   and gives a precise score. We keep the top 3 by score.
                       ↓
7. Safety check: is the closest of those 3 chunks close enough to
   the question? Cosine distance is 0.67 — under our 0.75 threshold,
   so yes. Continue.
                       ↓
8. Build the prompt: paste the 3 chunks tagged as [doc_0], [doc_1],
   [doc_2] into a template with strict instructions.
                       ↓
9. Send the prompt to Qwen via Ollama. Qwen returns:
   "The Academy of Motion Picture Arts and Sciences (AMPAS) [doc_0]
    is an organization that administers the Academy Awards…"
                       ↓
10. Check for [doc_N] citations. Found "[doc_0]". Mark confidence as
    "high".
                       ↓
11. Return JSON: {mode, question, answer, sources, confidence}
                       ↓
12. Browser renders a green-tagged bubble.
```

Total time on a cold cache: about 5 seconds. On a warm cache
(same question asked before): about 10 milliseconds.

---

## Why we made three key decisions

**Why keep everything local.** No cloud APIs means no cost per
question, no privacy leaks, and no internet dependency during a
demo. The tradeoff is that the local model (Qwen 7B) is smaller
than a cloud model like Gemini, so its answers are a bit rougher.
For a course demo, the local trade is worth it.

**Why put the reranker between Weaviate and the LLM.** Vector
search is fast but coarse — it can retrieve semantically close
chunks that aren't actually relevant. The cross-encoder reads each
`(question, chunk)` pair together and can tell when a chunk is a
distractor. Adding this step is the single easiest way to make
retrieval noticeably better.

**Why abstain when retrieval is weak.** A language model asked to
answer from irrelevant context will make something up — that's
its default behavior. No amount of "please don't hallucinate" in
the prompt fixes this reliably. The only way to *guarantee* no
hallucination on off-topic questions is to not call the model at
all. So we check the distance before calling, and if the closest
chunk is too far, we return a canned refusal.

---

## Summary in one paragraph

The app is a single Flask program that answers questions two
different ways: **Graph RAG** translates your question into a
database query and reads the answer from an RDF graph, while
**Text RAG** finds the most relevant paragraphs in a text corpus
and asks a language model to write an answer citing those
paragraphs. Both use the same local LLM (Qwen via Ollama), but the
Text RAG pipeline has extra safety layers — a smarter reranker
step, a "don't answer if retrieval failed" guardrail, and a
citation check that colors the answer bubble red, yellow, or green
depending on how confident we are. Everything runs on your laptop.
