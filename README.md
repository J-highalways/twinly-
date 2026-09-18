# twinly

Built for **Track 1: The Glass Box Problem** — Epochesque 2.0

## What this is

An AI system (agent / RAG app / LLM pipeline) built to be fully transparent — every step, decision, and cost can be traced and replayed, with no black-box behavior.

## Key features

- **Full observability** — every LLM call, tool call, and pipeline step is logged and traceable, so any run can be replayed step by step.
- **Context engineering** — keeps long conversations (tested across 15–20+ turns) accurate and on-topic through summarizing/trimming/selective retrieval.
- **Cost tracking** — token usage and cost are tracked and shown per run.
- **Caught failure case** — includes a real failure (e.g. wrong tool call / hallucination / context overflow) caught via tracing, along with the fix.

## Tech stack

- Google's Gemini-2.3-flash
- python
