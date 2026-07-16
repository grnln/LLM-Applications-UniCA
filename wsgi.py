"""Flask app: serves the chat UI on / and exposes both RAG endpoints."""
import os
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from src.graph_rag import build_chain, last_sparql
from src.graph_rag import warmup as _graph_warmup
from src.text_rag import ask as text_ask
from src.text_rag import warmup as _text_warmup

application = Flask(__name__, static_url_path='/static')

# Build the Graph RAG chain once at boot so requests don't pay startup cost.
_graph_chain = build_chain()

# Preload the LLM into Ollama's memory + SentenceTransformer + Weaviate.
# First real request is now warm instead of cold (saves ~5s on first hit).
_graph_warmup()
_text_warmup()


@lru_cache(maxsize=128)
def _run_graph(question: str) -> tuple[str, str | None]:
    """Run the Graph RAG chain and return (answer, sparql). Cached per question."""
    last_sparql.set(None)
    result = _graph_chain.invoke({_graph_chain.input_key: question})
    return result[_graph_chain.output_key], last_sparql.get()


def _get_question() -> str:
    """Extract and trim the 'question' field from the JSON body of the current request."""
    payload = request.get_json(silent=True) or {}
    return (payload.get('question') or '').strip()


@application.route('/')
def home() -> str:
    """Serve the chat UI (Jinja template)."""
    return render_template('home.html')


@application.get('/health')
def health():
    """Simple liveness check for smoke tests and monitoring."""
    return {'status': 'ok'}


@application.post('/ask/graph')
def ask_graph():
    """Graph RAG endpoint. Body: {question}. Returns {mode, question, answer, sparql}."""
    question = _get_question()
    if not question:
        return jsonify(error="missing 'question'"), 400
    try:
        answer, sparql = _run_graph(question)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f'{type(exc).__name__}: {exc}'), 502
    return jsonify(
        mode='graph',
        question=question,
        answer=answer,
        sparql=sparql,
    )


@application.post('/ask/text')
def ask_text():
    """Text RAG endpoint. Body: {question}. Returns {mode, question, answer, sources}."""
    question = _get_question()
    if not question:
        return jsonify(error="missing 'question'"), 400
    try:
        result = text_ask(question)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f'{type(exc).__name__}: {exc}'), 502
    return jsonify(
        mode='text',
        question=question,
        answer=result['answer'],
        sources=result['sources'],
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
