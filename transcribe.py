"""
Usage:
    modal run transcribe.py --file-id <google_drive_file_id> [--language ru|he|auto] [--output auto|<name>.txt]

Example:
    modal run transcribe.py --file-id <google_drive_file_id>

By default (--output auto) the transcript is saved under the file's original
Google Drive name with a .txt suffix. Pass --output <name>.txt to override.
"""

import modal
import os
import sys

# ── Modal image with all dependencies ────────────────────────────────────────
def _download_whisper_model():
    """Download whisper model at image build time so it's cached in the image."""
    import whisper
    whisper.load_model("medium")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "openai-whisper",
        "gdown",
        "pyannote.audio==3.3.2",
        "huggingface_hub<0.25",
        "torch>=2.1,<2.6",
        "torchaudio>=2.1,<2.6",
        "numpy",
        "soundfile",
        "matplotlib",
    )
    .run_function(_download_whisper_model)
)

app = modal.App("transcribe-audio", image=image)

# ── Local directories ─────────────────────────────────────────────────────────
AUDIO_DIR = "audio"              # downloaded intermediate audio + cached whisper results
TRANSCRIPTS_DIR = "transcripts"  # output .txt transcripts

# ── Secrets — set these in Modal dashboard or via `modal secret create` ──────
#   HF_TOKEN: HuggingFace token (required for pyannote speaker diarization)
#   Get one at https://huggingface.co/settings/tokens
#   Also accept the model terms at:
#     https://huggingface.co/pyannote/speaker-diarization-3.1
#     https://huggingface.co/pyannote/segmentation-3.0
secrets = [modal.Secret.from_name("huggingface-secret")]


# ── Step 1: Whisper transcription (the expensive part) ────────────────────────
@app.function(
    gpu="T4",
    timeout=600,
)
def transcribe(audio_bytes: bytes, language: str = "auto") -> dict:
    import tempfile
    import whisper

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    print("[whisper] Loading model...")
    model = whisper.load_model("medium")

    whisper_opts = dict(word_timestamps=True, task="transcribe")
    if language != "auto":
        whisper_opts["language"] = language

    print(f"[whisper] Transcribing... (language={language})")
    result = model.transcribe(audio_path, **whisper_opts)

    detected_lang = result.get("language", "unknown")
    print(f"[whisper] Detected language: {detected_lang}")

    # Return only the serialisable parts we need downstream
    return {
        "language": detected_lang,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ],
    }


# ── Step 2: Speaker diarization ──────────────────────────────────────────────
@app.function(
    gpu="T4",
    timeout=600,
    secrets=secrets,
)
def diarize(audio_bytes: bytes, whisper_result: dict) -> str:
    import tempfile
    import torch
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[diarization] No HF_TOKEN found — skipping speaker detection")
        return _format_without_speakers(whisper_result)

    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        f.write(audio_bytes)
        m4a_path = f.name

    # Convert to WAV — libsndfile (used by pyannote) can't read m4a
    wav_path = m4a_path.replace(".m4a", ".wav")
    print("[diarization] Converting m4a → wav...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", m4a_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True, capture_output=True,
    )
    audio_path = wav_path

    print("[diarization] Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(audio_path)

    return _merge_transcript_with_speakers(whisper_result["segments"], diarization)


def _format_without_speakers(result: dict) -> str:
    """Fallback: plain transcript without speaker labels."""
    return " ".join(seg["text"].strip() for seg in result["segments"])


def _merge_transcript_with_speakers(segments, diarization) -> str:
    """Assign each whisper segment a speaker label from pyannote."""

    # Build list of (start, end, speaker) from diarization
    speaker_turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    def get_speaker(seg_start: float, seg_end: float) -> str:
        """Return the speaker whose turn overlaps most with this segment."""
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for (t_start, t_end, speaker) in speaker_turns:
            overlap = max(0, min(seg_end, t_end) - max(seg_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        return best_speaker

    blocks = []
    prev_speaker = None
    current_texts = []

    for seg in segments:
        speaker = get_speaker(seg["start"], seg["end"])
        text = seg["text"].strip()
        if not text:
            continue

        if speaker != prev_speaker:
            if prev_speaker is not None and current_texts:
                blocks.append(f"{prev_speaker}: {' '.join(current_texts)}")
            prev_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if prev_speaker is not None and current_texts:
        blocks.append(f"{prev_speaker}: {' '.join(current_texts)}")

    return "\n".join(blocks)


def _resolve_drive_filename(file_id: str):
    """Best-effort fetch of the file's original name from Google Drive.

    Reads the Content-Disposition header from the Drive download endpoint with a
    1-byte range request, so it costs ~nothing and never downloads the file.
    Returns the original filename (str) or None if it can't be determined.
    """
    import re
    import urllib.parse
    import urllib.request

    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_disposition = resp.headers.get("Content-Disposition", "") or ""
    except Exception as e:
        print(f"[name] Could not resolve original filename: {e}")
        return None

    # RFC 5987 form: filename*=UTF-8''<percent-encoded>
    m = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", content_disposition, re.IGNORECASE)
    if m:
        return urllib.parse.unquote(m.group(1).strip().strip('"'))

    # Plain form: filename="..." — http.client decodes header bytes as latin-1,
    # so re-encode to recover the original UTF-8 name.
    m = re.search(r'filename="?([^";]+)"?', content_disposition)
    if m:
        name = m.group(1)
        try:
            name = name.encode("iso-8859-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        return name

    return None


# ── Entry point ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    file_id: str,
    language: str = "auto",       # "auto", "ru", "he"
    output: str = "auto",          # "auto" → derive from the Drive filename
    skip_whisper: bool = False,    # reuse cached whisper result
):
    import json
    import gdown

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

    cache_path = os.path.join(AUDIO_DIR, f"whisper_cache_{file_id}.json")
    audio_path = os.path.join(AUDIO_DIR, f"recording_{file_id}.m4a")

    # ── Resolve output filename ──────────────────────────────────────────────
    if output == "auto":
        original = _resolve_drive_filename(file_id)
        if original:
            output = os.path.splitext(os.path.basename(original))[0] + ".txt"
            print(f"[name] Output filename derived from Drive: {output}")
        else:
            output = f"transcript_{file_id}.txt"
            print(f"[name] Could not resolve Drive filename; using {output}")

    # A bare filename lands in TRANSCRIPTS_DIR; an explicit path is respected.
    output_path = output if os.path.dirname(output) else os.path.join(TRANSCRIPTS_DIR, output)

    # ── Step 1: Whisper (skip if cached) ─────────────────────────────────────
    if skip_whisper:
        print(f"[cache] Loading cached Whisper result from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            whisper_result = json.load(f)
    else:
        print(f"[download] Downloading from Google Drive (id={file_id})...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, audio_path, quiet=False)

        print("[upload] Sending audio to Modal for transcription...")
        audio_bytes = open(audio_path, "rb").read()
        whisper_result = transcribe.remote(audio_bytes, language=language)

        # Cache the Whisper result locally
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(whisper_result, f, ensure_ascii=False, indent=2)
        print(f"[cache] Whisper result saved to {cache_path}")

    # ── Step 2: Diarization ──────────────────────────────────────────────────
    audio_bytes = open(audio_path, "rb").read()

    print("[upload] Sending audio to Modal for diarization...")
    transcript = diarize.remote(audio_bytes, whisper_result)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"\n{'─'*60}")
    print(transcript)
    print(f"{'─'*60}")
    print(f"\n✅ Saved to {output_path}")
