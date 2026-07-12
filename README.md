# AI Reel Agent (MVP)

Podcast video in -> one vertical 9:16 reel out, with silence removed,
speaker-centered crop, and burned-in word-synced captions.

## Pipeline
```
input.mp4
  -> transcribe (faster-whisper, word-level timestamps)
  -> local LLM picks the best ~45s segment (Ollama)
  -> cut internal silence gaps
  -> detect speaker, crop to 9:16
  -> burn captions
  -> reel_output.mp4
```

## Setup

1. **ffmpeg** must be on your PATH.
   - Windows: download from https://www.gyan.dev/ffmpeg/builds/, add `bin/` to PATH.
   - Check with: `ffmpeg -version`

2. **Ollama** (local LLM runtime): https://ollama.com
   ```
   ollama pull llama3.1:8b
   ollama serve          # usually starts automatically after install
   ```
   Swap to a smaller model (e.g. `llama3.2:3b`, `qwen2.5:7b-instruct`) if 8B is
   too slow/heavy for your machine — pass it via `--llm-model`.

3. **Python deps**
   ```
   pip install -r requirements.txt
   ```
   First run will also download the Whisper model weights (needs internet once).

## Run

```
python main.py podcast.mp4 -o reel.mp4
```

Options:
```
--whisper-model    tiny|base|small|medium|large-v3   (default: base)
--llm-model        any Ollama model tag               (default: llama3.1:8b)
--target-duration  target clip length in seconds      (default: 45)
--silence-gap      cut gaps longer than this (sec)    (default: 0.5)
--words-per-caption words shown per caption line       (default: 4)
--keep-work-dir    keep transcript.json + intermediates for debugging
```

## Known MVP limitations (next things to fix)
- Crop is a single static x-offset, not dynamic panning — fine for one
  speaker sitting mostly still, will clip a speaker who moves a lot.
- Only handles a single speaker/face (largest detected face wins).
- Picks exactly one clip per run — no multi-clip / batch mode yet.
- Haar cascade face detection is fast but not robust to side profiles or
  low light. Swap in `mediapipe` or an OpenCV DNN face detector if this
  becomes a problem.
- No retry/self-review loop (agent doesn't watch its own output yet).

## Suggested next steps
1. Wrap this script's steps as LangGraph nodes (you already have the pattern
   from your other project) so you get retries, branching, and can swap in
   multi-clip mode later.
2. Add a vision-model review pass: sample frames of the final render, ask a
   local vision model "is the speaker in frame?" and re-crop if not.
3. Add caption style presets (word-highlight karaoke effect vs static line).
