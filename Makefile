# ── Transcribe Audio via Modal ────────────────────────────────────────────────
# Usage:
#   make run FILE_ID=<google_drive_file_id>
#   make run FILE_ID=<id> LANGUAGE=ru
#   make rerun FILE_ID=<id>              # skip whisper, reuse cached result

FILE_ID   ?=
LANGUAGE  ?= auto
OUTPUT    ?= transcript.txt

.PHONY: run rerun clean

# Full run: download → whisper → diarize
run:
	@test -n "$(FILE_ID)" || (echo "Error: FILE_ID is required. Usage: make run FILE_ID=<id>" && exit 1)
	modal run transcribe.py --file-id $(FILE_ID) --language $(LANGUAGE) --output $(OUTPUT)

# Rerun: skip whisper, use cached result → diarize only
rerun:
	@test -n "$(FILE_ID)" || (echo "Error: FILE_ID is required. Usage: make rerun FILE_ID=<id>" && exit 1)
	modal run transcribe.py --file-id $(FILE_ID) --language $(LANGUAGE) --output $(OUTPUT) --skip-whisper

# Remove cached whisper results
clean:
	rm -f whisper_cache_*.json
