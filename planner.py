"""
Ask a local LLM (via Ollama) to pick the best segment(s) of the transcript for
vertical reels. Requires `ollama serve` running locally and the target model
already pulled (e.g. `ollama pull llama3.1:8b`).
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SINGLE_PROMPT = """You are a short-form video editor selecting a clip for a vertical reel.

Below is a podcast transcript with [start-end] timestamps in seconds for each segment.
Pick ONE continuous span of roughly {target_duration:.0f} seconds (+/- 15s) that is the most
engaging, self-contained clip: it should have a hook, a complete thought, and should not start
or end mid-sentence.
{avoid_section}
Transcript:
{transcript_text}

Respond with ONLY valid JSON and nothing else, in exactly this shape:
{{"start": <number, seconds>, "end": <number, seconds>, "reason": "<one short sentence>"}}
"""


def _format_transcript(segments: list) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'].strip()}" for s in segments)


def _format_avoid(avoid_ranges: list) -> str:
    if not avoid_ranges:
        return ""
    spans = "; ".join(f"{s:.1f}-{e:.1f}s" for s, e in avoid_ranges)
    return f"\nDo NOT pick a span that overlaps any of these already-used ranges: {spans}\n"


def _estimate_tokens(text: str) -> int:
    # Rough heuristic for English text: ~1.3 tokens per whitespace-split word.
    return int(len(text.split()) * 1.3)


def _size_context(prompt: str, floor: int = 4096, ceiling: int = 32768) -> int:
    """Pick a num_ctx big enough to hold the whole prompt + response, sized to
    the nearest power of two so Ollama doesn't have to reallocate awkwardly."""
    needed = _estimate_tokens(prompt) + 1024  # headroom for the JSON response
    n = floor
    while n < needed and n < ceiling:
        n *= 2
    return min(n, ceiling)


def _closest_segment_end(segments: list, target_time: float) -> float:
    """Snap a clamp point to the end of the nearest real transcript segment
    instead of cutting mid-sentence at an arbitrary second."""
    earlier = [s["end"] for s in segments if s["end"] <= target_time]
    if earlier:
        return max(earlier)
    later = [s["end"] for s in segments if s["end"] > target_time]
    return min(later) if later else target_time


def _clamp(start: float, end: float, target_duration: float, segments: list) -> float:
    """If a plan wildly overshoots the requested duration (context-window
    truncation, model ignoring instructions, etc.), pull `end` back in --
    snapped to a sentence boundary so it doesn't cut mid-word."""
    if end - start <= target_duration * 2.5:
        return end
    snapped = _closest_segment_end(segments, start + target_duration)
    return snapped if snapped > start else start + target_duration


def _call_ollama(prompt: str, model: str, timeout: int) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": _size_context(prompt)},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def plan_edit(segments: list, target_duration: float = 45.0, model: str = "llama3.1:8b",
              timeout: int = 120, avoid_ranges: list = None) -> dict:
    prompt = SINGLE_PROMPT.format(target_duration=target_duration,
                                   avoid_section=_format_avoid(avoid_ranges),
                                   transcript_text=_format_transcript(segments))
    raw = _call_ollama(prompt, model, timeout)

    try:
        plan = json.loads(raw)
        start, end = float(plan["start"]), float(plan["end"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"LLM did not return a usable plan. Raw output:\n{raw}") from e

    if end <= start:
        raise RuntimeError(f"LLM returned an invalid range: start={start}, end={end}")

    end = _clamp(start, end, target_duration, segments)
    return {"start": start, "end": end, "reason": plan.get("reason", "")}


def plan_multi_clip(segments: list, num_clips: int = 3, target_duration: float = 45.0,
                     model: str = "llama3.1:8b", timeout: int = 120,
                     max_attempts_per_clip: int = 4) -> list:
    """
    Sequential single-clip calls instead of one big array request -- small
    local models are unreliable at returning the right top-level JSON shape
    (array vs object) when asked for N clips at once, so we ask for one clip
    at a time and tell the model which ranges are already taken.
    """
    chosen = []
    for _ in range(num_clips):
        avoid = [(c["start"], c["end"]) for c in chosen]
        picked = None
        for _attempt in range(max_attempts_per_clip):
            try:
                candidate = plan_edit(segments, target_duration=target_duration,
                                       model=model, timeout=timeout, avoid_ranges=avoid)
            except RuntimeError:
                continue  # bad JSON / invalid range this attempt -- just retry
            if not any(_overlaps(candidate, c) for c in chosen):
                picked = candidate
                break
            avoid = avoid + [(candidate["start"], candidate["end"])]  # tell it to avoid this miss too
        if picked:
            chosen.append(picked)

    if not chosen:
        raise RuntimeError("Could not obtain any valid non-overlapping clips from the LLM.")

    chosen.sort(key=lambda c: c["start"])
    return chosen