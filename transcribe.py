"""
Transcribe a video/audio file with word-level timestamps using faster-whisper.
faster-whisper decodes audio via PyAV internally, so you can pass a video file directly.
"""
from faster_whisper import WhisperModel


def transcribe(media_path: str, model_size: str = "base", device: str = "cpu",
                compute_type: str = "int8") -> dict:
    """
    Returns:
        {
          "language": str,
          "segments": [{"start": float, "end": float, "text": str}, ...],
          "words":    [{"start": float, "end": float, "word": str}, ...],
        }
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(media_path, word_timestamps=True)

    segments = []
    words = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})

    if not words:
        raise RuntimeError(
            "No word-level timestamps returned. Check the input has speech / a valid audio track."
        )

    return {"language": info.language, "segments": segments, "words": words}
