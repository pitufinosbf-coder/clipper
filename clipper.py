"""
Clipper: convierte un vídeo largo en clips verticales (9:16) con subtítulos grandes.

Uso:
  python clipper.py video.mp4 --clips 5
  python clipper.py video.mp4 --times "1:20-2:05, 10:00-10:45"
"""
import argparse
import json
import os
import re
import subprocess
import sys

OUT_W, OUT_H = 1080, 1920


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def duration_of(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def parse_time(t):
    parts = [float(p) for p in t.strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ---------- Transcripción ----------

def transcribe(path, model_size, language):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        path, language=None if language == "auto" else language,
        word_timestamps=True, vad_filter=True)
    print(f"Idioma detectado: {info.language}", flush=True)
    segs = []
    for s in segments:
        words = [{"start": w.start, "end": w.end, "word": w.word.strip()} for w in (s.words or []) if w.word.strip()]
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": words})
        print(f"  [{fmt_time(s.start)}] {s.text.strip()}", flush=True)
    return segs


# ---------- Volumen por segundo (para detectar momentos intensos) ----------

def loudness_per_second(path):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-ac", "1",
         "-af", "aresample=16000,asetnsamples=n=16000,astats=metadata=1:reset=1,"
                "ametadata=print:key=lavfi.astats.Overall.RMS_level",
         "-f", "null", "-"],
        capture_output=True, text=True)
    vals = []
    for m in re.finditer(r"RMS_level=(-?[\d.]+|-inf)", proc.stderr):
        v = m.group(1)
        vals.append(-90.0 if v == "-inf" else float(v))
    return vals


# ---------- Elegir los mejores momentos ----------

HOOK_WORDS = ("nunca", "nadie", "secreto", "verdad", "increíble", "brutal", "por qué", "cómo",
              "never", "nobody", "secret", "truth", "crazy", "insane", "why", "how")


def score_window(segs, loud, start, end):
    text = " ".join(s["text"] for s in segs).lower()
    n_words = sum(len(s["words"]) or len(s["text"].split()) for s in segs)
    dur = max(end - start, 1)
    pace = n_words / dur  # palabras por segundo
    window = loud[int(start):int(end) + 1] or [-60]
    energy = sum(window) / len(window)
    peak = max(window)
    first = segs[0]["text"].lower()
    hook = 0.0
    if "?" in first or "¿" in first:
        hook += 1.0
    if "!" in text or "¡" in text:
        hook += 0.5
    hook += 0.5 * sum(1 for w in HOOK_WORDS if w in first)
    return pace * 1.5 + (energy + 60) / 10 + (peak + 60) / 20 + hook


def pick_moments(segs, loud, n, min_len, max_len):
    candidates = []
    for i in range(len(segs)):
        start = segs[i]["start"]
        group = []
        for j in range(i, len(segs)):
            group.append(segs[j])
            length = segs[j]["end"] - start
            if length > max_len:
                break
            if length >= min_len:
                candidates.append((score_window(group, loud, start, segs[j]["end"]), start, segs[j]["end"], list(group)))
    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen = []
    for c in candidates:
        if all(c[2] <= o[1] or c[1] >= o[2] for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c[1])


# ---------- Subtítulos estilo TikTok ----------

def ass_time(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def build_ass(words, clip_start, clip_end, path, per_line=3):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,86,&H00FFFFFF,&H0000D7FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    inside = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    lines = []
    for k in range(0, len(inside), per_line):
        chunk = inside[k:k + per_line]
        s = chunk[0]["start"] - clip_start
        e = (inside[k + per_line]["start"] if k + per_line < len(inside) else chunk[-1]["end"]) - clip_start
        e = min(e, chunk[-1]["end"] - clip_start + 0.4)
        # la palabra que suena se resalta en amarillo
        for idx, w in enumerate(chunk):
            ws = w["start"] - clip_start
            we = chunk[idx + 1]["start"] - clip_start if idx + 1 < len(chunk) else e
            parts = []
            for jdx, x in enumerate(chunk):
                t = x["word"].upper().replace("{", "").replace("}", "")
                parts.append("{\\c&H00D7FF&}" + t + "{\\c&HFFFFFF&}" if jdx == idx else t)
            lines.append(f"Dialogue: 0,{ass_time(max(ws, s))},{ass_time(we)},Cap,,0,0,0,,{' '.join(parts)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


# ---------- Render ----------

def render(src, start, end, ass_path, out_path, mode):
    if mode == "crop":
        vf = f"scale=-2:{OUT_H},crop={OUT_W}:{OUT_H}"
    else:  # blur: vídeo entero en el centro y fondo desenfocado
        vf = (f"split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
              f"crop={OUT_W}:{OUT_H},boxblur=25:5[bg];[b]scale={OUT_W}:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if ass_path:
        vf += f",ass={ass_path}"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", src,
         "-filter_complex", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--clips", type=int, default=5, help="cuántos clips sacar en modo automático")
    ap.add_argument("--min", type=float, default=25, help="duración mínima de cada clip (s)")
    ap.add_argument("--max", type=float, default=60, help="duración máxima de cada clip (s)")
    ap.add_argument("--times", default="", help='tramos manuales: "1:20-2:05, 10:00-10:45"')
    ap.add_argument("--mode", choices=["blur", "crop"], default="blur")
    ap.add_argument("--lang", default="auto", help="es, en o auto")
    ap.add_argument("--model", default="small", help="tiny, base, small, medium")
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--out", default="clips")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    segs = [] if a.no_subs and a.times else transcribe(a.video, a.model, a.lang)
    words = [w for s in segs for w in s["words"]]

    if a.times.strip():
        moments = []
        for part in a.times.split(","):
            s, e = part.split("-")
            st, en = parse_time(s), parse_time(e)
            moments.append((0, st, en, [x for x in segs if x["end"] > st and x["start"] < en]))
    else:
        if not segs:
            sys.exit("No se ha detectado voz en el vídeo. Usa --times para elegir los tramos a mano.")
        moments = pick_moments(segs, loudness_per_second(a.video), a.clips, a.min, a.max)

    summary = ["# Clips generados\n"]
    for n, (_, st, en, group) in enumerate(moments, 1):
        name = f"clip_{n:02d}"
        ass = None
        if not a.no_subs and words:
            ass = os.path.join(a.out, f"{name}.ass")
            build_ass(words, st, en, ass)
        render(a.video, st, en, ass, os.path.join(a.out, f"{name}.mp4"), a.mode)
        if ass:
            os.remove(ass)
        text = " ".join(s["text"] for s in group)
        summary.append(f"## {name}.mp4  ({fmt_time(st)} - {fmt_time(en)}, {en - st:.0f} s)\n\n"
                       f"**Gancho sugerido:** {group[0]['text'] if group else ''}\n\n{text}\n")
    with open(os.path.join(a.out, "LEEME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\nListo: {len(moments)} clips en '{a.out}/'")


if __name__ == "__main__":
    main()
