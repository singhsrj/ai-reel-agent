
# 🎬 AI Reel Agent (MVP)

> **Turn long-form podcasts into viral vertical clips automatically.**

AI Reel Agent is a fully local pipeline that takes a horizontal podcast video and automatically generates a ready-to-post 9:16 vertical reel. It finds the best moments, removes dead air, centers the speaker, and adds word-synced captions.

---

## ✨ Features

- **🧠 Smart Selection:** Uses local LLMs (via Ollama) to analyze the transcript and extract the most engaging ~45-second segment.
- **✂️ Auto-Editing:** Automatically cuts internal silence gaps to keep the pace energetic.
- **📱 Smart Crop:** Detects the speaker's face and crops the video to a vertical 9:16 aspect ratio.
- **💬 Dynamic Captions:** Burns in word-level, synced captions directly onto the video.
- **🔒 100% Local:** Runs entirely on your machine using Faster-Whisper and local LLMs.

---

## 🔄 How it Works

The pipeline processes your video in a single pass:

`input.mp4` 
 ⬇️ **Transcribe** (faster-whisper, word-level timestamps)
 ⬇️ **Select** (Ollama picks the best ~45s segment)
 ⬇️ **Trim** (Cut internal silence gaps)
 ⬇️ **Crop** (Detect speaker, crop to 9:16)
 ⬇️ **Caption** (Burn in word-synced text)
🎉 `reel_output.mp4`

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **FFmpeg:** Must be installed and accessible on your system `PATH`.
  - *Windows:* Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract, and add the `bin/` folder to your PATH.
  - *Verify:* Run `ffmpeg -version` in your terminal.
- **Ollama:** The local LLM runtime. Download from [ollama.com](https://ollama.com).

### 2. Prepare the LLM
Once Ollama is installed, pull the default model (or your preferred alternative):
```bash
ollama pull llama3.1:8b
ollama serve  # usually starts automatically after install

```

*💡 Tip: If 8B is too heavy for your machine, swap to a smaller model like `llama3.2:3b` or `qwen2.5:7b-instruct` using the `--llm-model` flag.*

### 3. Install Dependencies

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt

```

*(Note: The first run will download the Whisper model weights, which requires an internet connection.)*

---

## 💻 Usage

Run the script on your target video:

```bash
python main.py podcast.mp4 -o reel.mp4

```

### ⚙️ Configuration Options

| Option | Description | Default |
| --- | --- | --- |
| `--whisper-model` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `--llm-model` | Any installed Ollama model tag | `llama3.1:8b` |
| `--target-duration` | Target clip length in seconds | `45` |
| `--silence-gap` | Cut audio gaps longer than this (in seconds) | `0.5` |
| `--words-per-caption` | Number of words shown per caption line | `4` |
| `--keep-work-dir` | Keep `transcript.json` and intermediates for debugging | *Disabled* |

---

## 🚧 Known Limitations & Roadmap

This is an MVP. Here is what is currently being worked on or planned for future releases:

### Current Limitations

* **Static Cropping:** The crop is a single static x-offset. It works fine for one speaker sitting mostly still, but will clip a speaker who moves around significantly.
* **Single Subject:** Currently only handles a single speaker/face (the largest detected face wins).
* **One Clip Per Run:** Picks exactly one clip per run. Multi-clip or batch processing is not yet supported.
* **Basic Face Detection:** Uses Haar cascades for speed, which can struggle with side profiles or low light.
* **No Self-Correction:** The agent does not watch or review its own final output.

### 🎯 Next Steps / To-Do

* [ ] **LangGraph Integration:** Wrap pipeline steps as LangGraph nodes to introduce retries, branching logic, and multi-clip support.
* [ ] **Vision-Model Review Pass:** Sample frames of the final render and ask a local vision model, *"is the speaker in frame?"*, triggering a re-crop if necessary.
* [ ] **Advanced Face Tracking:** Swap in `mediapipe` or an OpenCV DNN face detector for more robust tracking.
* [ ] **Caption Styling:** Add aesthetic presets (e.g., word-highlight karaoke effect vs. static lines).
