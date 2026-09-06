# speaker-transcripts

Turn a recording into a speaker-labeled transcript — who said what, not just what was said.

Runs OpenAI Whisper (transcription) and pyannote.audio (speaker diarization) on GPU
via [Modal](https://modal.com), then merges the two into a single readable transcript.

```
SPEAKER_04: Так это после голодания.
SPEAKER_11: А они теперь вторую готовят.
SPEAKER_07: Это точно. Я поэтому и говорю...
```

## Capabilities

- **Speaker attribution** — pyannote assigns each utterance a `SPEAKER_XX` label. The
  number of speakers is detected automatically; no need to declare it up front.
- **Automatic language detection** — Whisper detects the language, or pass a hint
  (`LANGUAGE=ru`) to skip detection and improve accuracy.
- **Multilingual** — anything Whisper supports. Tested on Russian and Hebrew.
- **Readable output** — consecutive segments from the same speaker are merged into one
  block, so you get paragraphs rather than a wall of timestamped fragments.
- **Google Drive input** — point it at a Drive file ID; the audio is fetched for you.
- **Auto-named output** — the transcript takes the original Drive filename, so
  `29 ОТЗЫВЫ о Фильме.m4a` becomes `29 ОТЗЫВЫ о Фильме.txt`.
- **Cached Whisper results** — the expensive transcription step is cached locally, so
  you can re-run diarization alone without paying for Whisper twice.
- **GPU on demand** — Modal spins up a T4 per run and bills only for time consumed.
  No always-on infrastructure.
- **Long-audio friendly** — functions allow up to 2h each; multi-hour recordings are fine.
- **Degrades gracefully** — without an `HF_TOKEN` it still produces a plain transcript,
  just without speaker labels.

## How it works

1. **Download** — fetch the audio from Google Drive by file ID.
2. **Transcribe** — Whisper (`medium`) on a T4, with word-level timestamps.
3. **Diarize** — pyannote `speaker-diarization-3.1` segments the audio by speaker.
4. **Merge** — each Whisper segment is assigned the speaker whose turn overlaps it most,
   then adjacent same-speaker segments are joined into blocks.

Whisper and pyannote run as separate Modal functions, which is why step 2 can be cached
and skipped independently of step 3.

## Prerequisites

- [Modal](https://modal.com) account with the `modal` CLI authenticated (`modal token new`)
- A Modal secret named `huggingface-secret` containing your `HF_TOKEN`
- Accept the pyannote model terms (required, else diarization 401s):
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0
- The Drive file must be shared as **"Anyone with the link"** — downloads are anonymous.

## Usage

```bash
# Full run (whisper + diarization). Output: "<original Drive name>.txt"
make run FILE_ID=<google_drive_file_id>

# With language hint — faster and more accurate than auto-detection
make run FILE_ID=<id> LANGUAGE=ru

# Override the auto-derived output name
make run FILE_ID=<id> LANGUAGE=ru OUTPUT=transcript-27.txt

# Re-run diarization only (skip whisper, reuse cached result)
make rerun FILE_ID=<id>

# Move finished transcripts and their audio to ./archive
make archive

# Remove cached whisper results
make clean
```

By default (`OUTPUT=auto`) the transcript is named after the file's original Google Drive
name with a `.txt` suffix. Pass `OUTPUT=<name>.txt` to override.

Or invoke Modal directly:

```bash
modal run transcribe.py --file-id <google_drive_file_id> \
  [--language ru|he|auto] [--output auto|<name>.txt] [--skip-whisper]
```

## Layout

| Path           | Contents                                            |
| -------------- | --------------------------------------------------- |
| `audio/`       | Downloaded audio + cached Whisper results (JSON)     |
| `transcripts/` | Finished `.txt` transcripts                          |
| `archive/`     | Where `make archive` moves completed work            |

## Notes

- **Speaker labels are per-run.** `SPEAKER_04` in one transcript has no relationship to
  `SPEAKER_04` in another — pyannote doesn't identify people across recordings.
- **Cost scales with audio length.** A multi-hour recording means a multi-hour T4 run.
- **Whisper model size** is set in `transcribe.py` (`medium`). Larger is more accurate
  and slower; smaller is the reverse.
