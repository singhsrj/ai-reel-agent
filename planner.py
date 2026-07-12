"""
Ask a local LLM (via Ollama) to pick the single best continuous segment of the
transcript for a short vertical reel. Requires `ollama serve` running locally
and the target model already pulled (e.g. `ollama pull llama3.1:8b`).
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are a short-form video editor selecting a clip for a vertical reel.

Below is a podcast transcript with [start-end] timestamps in seconds for each segment.
Pick ONE continuous span of roughly {target_duration:.0f} seconds (+/- 15s) that is the most
engaging, self-contained clip: it should have a hook, a complete thought, and should not start
or end mid-sentence.

Transcript:
{transcript_text}

Respond with ONLY valid JSON and nothing else, in exactly this shape:
{{"start": <number, seconds>, "end": <number, seconds>, "reason": "<one short sentence>"}}
"""


def plan_edit(segments: list, target_duration: float = 45.0, model: str = "llama3.1:8b",
              timeout: int = 120) -> dict:
    transcript_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'].strip()}" for s in segments
    )
    prompt = PROMPT_TEMPLATE.format(target_duration=target_duration, transcript_text=transcript_text)

    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]

    try:
        plan = json.loads(raw)
        start = float(plan["start"])
        end = float(plan["end"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"LLM did not return a usable plan. Raw output:\n{raw}") from e

    if end <= start:
        raise RuntimeError(f"LLM returned an invalid range: start={start}, end={end}")

    return {"start": start, "end": end, "reason": plan.get("reason", "")}
