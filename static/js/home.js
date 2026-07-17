/* Chat UI: send the question to the backend, render the reply. */

const chat = document.getElementById('chat');
const input = document.getElementById('prompt-input');
const button = document.getElementById('prompt-btn');

function bubble(text, side) {
    const el = document.createElement('div');
    el.className = `alert my-1 ${side === 'right' ? 'alert-secondary align-self-end' : 'alert-light align-self-start'}`;
    el.style.maxWidth = '75%';
    el.style.whiteSpace = 'pre-wrap';
    el.textContent = text;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
}

/* Attach a collapsible SPARQL block below an existing bubble. */
function attachSparql(bubbleEl, sparql) {
    const details = document.createElement('details');
    details.className = 'mt-2';
    const summary = document.createElement('summary');
    summary.className = 'text-muted small';
    summary.style.cursor = 'pointer';
    summary.textContent = 'Show generated SPARQL';
    const pre = document.createElement('pre');
    pre.className = 'small text-secondary mt-2 mb-0 p-2 bg-light border rounded';
    pre.style.whiteSpace = 'pre-wrap';
    pre.textContent = sparql;
    details.appendChild(summary);
    details.appendChild(pre);
    bubbleEl.appendChild(details);
}

/* Recolor the bubble and add a small confidence tag for Text RAG answers. */
function markConfidence(bubbleEl, confidence) {
    const tag = document.createElement('div');
    tag.className = 'small mt-2 fw-bold';
    if (confidence === 'abstain') {
        bubbleEl.classList.remove('alert-light');
        bubbleEl.classList.add('alert-danger');
        tag.textContent = 'Abstained — retrieval was too weak to answer safely';
    } else if (confidence === 'low') {
        bubbleEl.classList.remove('alert-light');
        bubbleEl.classList.add('alert-warning');
        tag.textContent = 'Low confidence — no source citations in this answer';
    } else if (confidence === 'high') {
        tag.textContent = 'Grounded — answer cites retrieved passages';
        tag.classList.add('text-success');
    } else {
        return;
    }
    bubbleEl.appendChild(tag);
}

async function submit() {
    const question = input.value.trim();
    if (!question) return;
    const mode = document.querySelector('input[name="mode"]:checked').value;

    bubble(question, 'right');
    const reply = bubble('...', 'left');
    input.value = '';
    button.disabled = true;

    try {
        const res = await fetch(`/ask/${mode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await res.json();
        reply.textContent = data.answer || data.error || '(no reply)';
        if (mode === 'graph' && data.sparql) {
            attachSparql(reply, data.sparql);
        }
        if (mode === 'text' && data.confidence) {
            markConfidence(reply, data.confidence);
        }
    } catch (err) {
        reply.textContent = `Error: ${err.message}`;
    } finally {
        button.disabled = false;
    }
}

button.addEventListener('click', submit);
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit();
});
