import os
import glob
import json
import time
import uuid
import requests
import chromadb
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# NOTE: this key is hardcoded so the script runs with no extra setup, as
# requested. It was pasted in plain text earlier in this project's chat
# history, so treat it as already public — rotate it in Google AI Studio
# whenever you get a chance, and swap the new one in below.
API_KEY = "AQ.Ab8RN6IPHcrzeo46HisEqe-iuGrptSmRY_RR92jvQ7522121QA"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# CONTEXT ENGINEERING CONSTANTS
# ---------------------------------------------------------------------------
SUMMARY_TRIGGER_TOKENS = 3000
MIN_RAW_MESSAGES = 4

FALLBACK_MESSAGE = "That's not something I can answer from this document — try asking something related to it. 🎀"


def log(run_id, step_type, data):
    event = {
        "run_id": run_id,
        "timestamp": time.time(),
        "step_type": step_type,
        "data": data
    }

    os.makedirs("logs", exist_ok=True)

    with open(f"logs/{run_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def load_documents():
    os.makedirs(DATA_DIR, exist_ok=True)

    texts = []

    for path in glob.glob(os.path.join(DATA_DIR, "*")):
        if os.path.isdir(path):
            continue

        try:
            if path.lower().endswith(".pdf"):
                reader = PdfReader(path)
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            print(f"[WARN] Skipping {path}: could not read file ({e})")
            continue

        if not text.strip():
            print(f"[WARN] {path} produced no extractable text — skipping.")
            continue

        texts.append((path, text))

    if not texts:
        print(
            f"[WARN] No readable files found in '{DATA_DIR}/'. "
            "Add a .pdf or .txt file there before asking questions."
        )

    return texts


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection("my_docs")


def build_database(force=False):
    """
    Rebuilds the vector DB from scratch every time it's called.
    `force=False` (default) skips rebuilding if the collection already
    has data, so repeated runs don't error out or waste time re-embedding.
    Pass force=True to always wipe and re-index (e.g. after editing data/).
    """
    global collection

    if not force and collection.count() > 0:
        print(f"Using existing index ({collection.count()} chunks). "
              f"Call build_database(force=True) to rebuild from scratch.")
        return collection.count()

    # Wipe any existing entries so we never collide on chunk ids.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    docs = load_documents()
    idx = 0

    for path, text in docs:
        chunks = chunk_text(text)

        for chunk in chunks:
            collection.add(
                documents=[chunk],
                ids=[f"chunk-{idx}"],
                metadatas=[{"source": path}]
            )

            idx += 1

    print(f"Indexed {idx} chunks from {len(docs)} files.")

    return idx


def call_gemini(prompt):
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            GEMINI_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": API_KEY
            },
            json=body,
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        return (f"ERROR: request failed ({e})", 0, 0, 0)

    if response.status_code != 200:
        return (
            f"ERROR: Gemini API returned status {response.status_code}: {response.text[:500]}",
            0, 0, 0
        )

    try:
        data = response.json()
    except ValueError:
        return (f"ERROR: could not parse response as JSON: {response.text[:500]}", 0, 0, 0)

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]

        usage = data.get("usageMetadata", {})

        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        return (
            answer,
            input_tokens,
            output_tokens,
            total_tokens
        )

    except (KeyError, IndexError):
        return (
            f"ERROR: unexpected response shape: {data}",
            0,
            0,
            0
        )


# ---------------------------------------------------------------------------
# CONTEXT ENGINEERING: token estimate + rolling summary
# ---------------------------------------------------------------------------

def estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)


def _messages_text(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def block_tokens(messages, summary=""):
    return estimate_tokens(_messages_text(messages)) + estimate_tokens(summary)


def summarize_history(run_id, existing_summary, turns_to_fold):
    turns_text = _messages_text(turns_to_fold)

    prompt = f"""Summarize the earlier parts of this conversation in 2-3 short sentences. Keep any facts or specifics the user gave that might matter later.

Previous summary (may be empty): {existing_summary}

New turns to fold in: {turns_text}

Updated summary:"""

    summary, input_tokens, output_tokens, total_tokens = call_gemini(prompt)

    if summary.startswith("ERROR:"):
        # Don't let a failed summarization call wipe out the existing summary.
        log(run_id, "context_summarization_error", {"error": summary})
        return existing_summary, 0.0

    cost = (input_tokens / 1_000_000) * 0.75 + (output_tokens / 1_000_000) * 3.75

    log(
        run_id,
        "context_summarization",
        {
            "folded_turns": len(turns_to_fold),
            "new_summary": summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 8)
        }
    )

    return summary.strip(), cost


def trim_and_summarize(run_id, prior_messages, context_summary, summarized_count):
    unsummarized = prior_messages[summarized_count:]
    tokens_before = block_tokens(unsummarized, context_summary)
    summarization_cost = 0.0
    summarized_this_turn = False

    while (
        len(unsummarized) > MIN_RAW_MESSAGES
        and block_tokens(unsummarized, context_summary) > SUMMARY_TRIGGER_TOKENS
    ):
        turn_to_fold = unsummarized[:2]
        unsummarized = unsummarized[2:]
        summarized_count += len(turn_to_fold)

        context_summary, fold_cost = summarize_history(
            run_id, context_summary, turn_to_fold
        )
        summarization_cost += fold_cost
        summarized_this_turn = True

    tokens_after = block_tokens(unsummarized, context_summary)

    log(
        run_id,
        "context_engineering",
        {
            "turns_before": len(prior_messages) // 2,
            "turns_kept_raw": len(unsummarized) // 2,
            "estimated_tokens_before": tokens_before,
            "estimated_tokens_after": tokens_after,
            "summarization_triggered": summarized_this_turn,
            "trigger_threshold_tokens": SUMMARY_TRIGGER_TOKENS
        }
    )

    return unsummarized, context_summary, summarized_count, summarization_cost


def ask(question, history=None, context_summary="", summarized_count=0):
    if history is None:
        history = []

    run_id = str(uuid.uuid4())[:8]

    log(
        run_id,
        "user_question",
        {
            "question": question,
            "turns_in_history": len(history) // 2
        }
    )

    # --- context engineering: decide what the model actually sees ---
    raw_recent, context_summary, summarized_count, summarization_cost = trim_and_summarize(
        run_id, history, context_summary, summarized_count
    )

    conversation_block = ""
    if context_summary:
        conversation_block += f"Summary of earlier conversation:\n{context_summary}\n\n"
    if raw_recent:
        conversation_block += "Recent conversation:\n" + _messages_text(raw_recent) + "\n\n"

    # --- RAG retrieval ---
    if collection.count() == 0:
        answer = FALLBACK_MESSAGE
        log(run_id, "retrieval", {"num_chunks": 0, "sources": [], "note": "collection empty"})
        print(f"\nANSWER: {answer}")
        print(f"(run_id: {run_id} | no documents indexed)")
        return (
            answer, 0, 0, 0, 0.0, 0.0, context_summary, summarized_count
        )

    results = collection.query(
        query_texts=[question],
        n_results=min(3, collection.count())
    )

    docs_found = results.get("documents", [[]])[0]
    metas_found = results.get("metadatas", [[]])[0]

    context = "\n\n".join(docs_found)
    sources = [m.get("source", "unknown") for m in metas_found]

    log(
        run_id,
        "retrieval",
        {
            "num_chunks": len(docs_found),
            "sources": sources
        }
    )

    prompt = f"""{conversation_block}Answer the question using only the context below. If the answer isn't in the context, say "That's not something I can answer from this document — try asking something related to it."

Context: {context}

Question: {question}
"""

    start_time = time.time()

    (
        answer,
        input_tokens,
        output_tokens,
        total_tokens
    ) = call_gemini(prompt)

    raw_answer = answer

    if answer.startswith("ERROR:"):
        print(f"\n[DEBUG] Real API error: {raw_answer}\n")
        answer = "Sorry, I hit an error talking to the model — check the terminal for details. 🎀"
    elif "i don't know" in answer.lower() or "not something i can answer" in answer.lower():
        answer = FALLBACK_MESSAGE

    latency = round(
        time.time() - start_time,
        2
    )

    input_cost = (
        input_tokens / 1000000
    ) * 0.75

    output_cost = (
        output_tokens / 1000000
    ) * 3.75

    estimated_cost = (
        input_cost + output_cost + summarization_cost
    )

    log(
        run_id,
        "llm_call",
        {
            "answer": answer,
            "raw_answer": raw_answer,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "answer_cost_usd": round(input_cost + output_cost, 8),
            "summarization_cost_usd": round(summarization_cost, 8),
            "estimated_cost_usd": round(estimated_cost, 8)
        }
    )

    print(f"\nANSWER: {answer}")
    print(
        f"(run_id: {run_id} | latency: {latency}s)"
    )
    print(
        f"Input tokens: {input_tokens}"
    )
    print(
        f"Output tokens: {output_tokens}"
    )
    print(
        f"Total tokens: {total_tokens}"
    )
    print(
        f"Estimated cost: ${estimated_cost:.8f}"
    )

    return (
        answer,
        input_tokens,
        output_tokens,
        total_tokens,
        estimated_cost,
        latency,
        context_summary,
        summarized_count
    )


if __name__ == "__main__":
    build_database()

    cli_history = []
    cli_summary = ""
    cli_summarized_count = 0

    while True:
        q = input(
            "\nAsk a question (or 'quit'): "
        )

        if q.lower() == "quit":
            break

        result = ask(q, cli_history, cli_summary, cli_summarized_count)

        answer = result[0]
        cli_summary = result[6]
        cli_summarized_count = result[7]

        cli_history.append({"role": "user", "content": q})
        cli_history.append({"role": "assistant", "content": answer})