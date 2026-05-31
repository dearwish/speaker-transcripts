# modal-transcribe-audio

Audio transcription + speaker diarization pipeline running on [Modal](https://modal.com).

Uses OpenAI Whisper (medium) for transcription and pyannote.audio for speaker diarization, both on GPU.

## Prerequisites

- [Modal](https://modal.com) account with `modal` CLI authenticated (`modal token new`)
- A Modal secret named `huggingface-secret` containing your `HF_TOKEN`
- Accept the pyannote model terms:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0

## Usage

```bash
# Full run (whisper + diarization). Output is saved as "<original Drive name>.txt"
make run FILE_ID=<google_drive_file_id>

# With language hint
make run FILE_ID=<id> LANGUAGE=ru

# Override the auto-derived output name
make run FILE_ID=<id> LANGUAGE=ru OUTPUT=transcript-27.txt

# Rerun diarization only (skip whisper, use cached result)
make rerun FILE_ID=<id>

# Move transcript files to ./archive
make archive

# Clean cached whisper results
make clean
```

By default (`OUTPUT=auto`) the transcript file is named after the file's original
Google Drive name with a `.txt` suffix. Pass `OUTPUT=<name>.txt` to override.

Or directly:

```bash
modal run transcribe.py --file-id <google_drive_file_id> [--language ru|he|auto] [--output auto|<name>.txt] [--skip-whisper]
```
