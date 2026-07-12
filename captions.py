"""
Build an .ass subtitle file from word timestamps, grouped a few words per
line, styled for vertical short-form video (large bold centered text, lower
portion of frame).
"""

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,84,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5,0,2,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def seconds_to_ass_time(t: float) -> str:
    t = max(t, 0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(words: list, out_path: str, words_per_line: int = 4) -> None:
    """
    `words` must already be on the FINAL (post-cut) timeline, i.e. remapped
    via silence.remap_time, with the clip starting at t=0.
    """
    lines = [ASS_HEADER]

    def flush(chunk):
        if not chunk:
            return
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        text = " ".join(c["word"].strip() for c in chunk if c["word"].strip()).upper()
        if text:
            lines.append(
                f"Dialogue: 0,{seconds_to_ass_time(start)},{seconds_to_ass_time(end)},Default,,0,0,0,,{text}"
            )

    chunk = []
    for w in words:
        chunk.append(w)
        if len(chunk) == words_per_line:
            flush(chunk)
            chunk = []
    flush(chunk)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
