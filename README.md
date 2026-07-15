# Oscar RAG — Graph RAG vs Text RAG

## The idea

A single web app that answers the **same natural-language questions** against
the **same domain** (the 98th Academy Awards, 2026) using **two different
Retrieval-Augmented Generation pipelines** side by side. The user picks
which pipeline to use with a radio button, so the comparison is direct.

We built the two datasets to be **complementary on purpose**: the RDF
knowledge graph holds every award outcome (winners, nominations, milestones)
but no film plots; the text corpus describes every film's plot and cast but
mentions no outcomes. Each pipeline shines on different questions, which
makes the demo tell a clear story about what each retrieval style is good
at.

## The architecture

```
Browser ─▶ Flask (wsgi.py) ─┬─▶ /ask/graph ─▶ Qwen writes SPARQL ─▶ GraphDB ─▶ Qwen formats answer
                            └─▶ /ask/text  ─▶ MiniLM embeds Q  ─▶ Weaviate top-K ─▶ Qwen answers
```

Two independent pipelines behind the same Flask app. Both use the **same
local LLM** (Qwen 2.5 Coder 7B via Ollama), but Graph RAG wraps it to force
clean SPARQL output, while Text RAG uses it plainly for prose generation.
Everything runs locally — no cloud API calls.

## The implementation

**Graph RAG** (`src/graph_rag.py`) uses LangChain's `OntotextGraphDBQAChain`.
The LLM sees the ontology (`ontology/oscar_schema.ttl`) as its "map of the
graph" and writes a SPARQL query, GraphDB executes it against the loaded
triples (`ontology/oscars2026.trig` + `award_labels.trig`), and the LLM
turns the rows into natural language. A custom `OllamaForSparql` subclass
pins the correct namespace prefix and strips markdown fences that small
models tend to add.

**Text RAG** (`src/text_rag.py` + `src/ingest_text.py`) uses classic dense
retrieval. At ingest, each paragraph of `ontology/oscar.txt` is embedded
with SentenceTransformer (MiniLM, 384 dims) and upserted into a Weaviate
collection. At query time, the question is embedded, Weaviate returns the
top-K nearest chunks, and the LLM answers using ONLY those chunks — with
citations like `[doc_3]` and an explicit "not-in-corpus" escape.

**Web app** (`wsgi.py` + `templates/` + `static/js/home.js`) is a single
Flask + Jinja + Bootstrap page with a chat UI, a mode toggle, and two
endpoints (`/ask/graph`, `/ask/text`) that both return the same JSON
envelope. **Infrastructure**: Weaviate runs in a Docker container defined
in `docker-compose.yml`; GraphDB runs natively; Ollama serves the LLM
locally.

## Setup

Full walkthrough (fresh laptop → working demo in ~30 min) is in
[`tutorial.md`](tutorial.md). Once set up, one command starts everything:

```bash
./run.sh          # http://localhost:5000
```
