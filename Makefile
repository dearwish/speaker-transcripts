# ── Transcribe Audio via Modal ────────────────────────────────────────────────
# Usage:
#   make run FILE_ID=<google_drive_file_id>
#   make run FILE_ID=<id> LANGUAGE=ru
#   make rerun FILE_ID=<id>              # skip whisper, reuse cached result

export PATH := $(shell python3 -c "import sys, os; print(os.path.dirname(sys.executable))"):$(PATH)

FILE_ID   ?=
LANGUAGE  ?= auto
OUTPUT    ?= transcript.txt

.PHONY: run rerun archive clean

# Full run: download → whisper → diarize
run:
	@test -n "$(FILE_ID)" || (echo "Error: FILE_ID is required. Usage: make run FILE_ID=<id>" && exit 1)
	modal run transcribe.py --file-id $(FILE_ID) --language $(LANGUAGE) --output $(OUTPUT)

# Rerun: skip whisper, use cached result → diarize only
rerun:
	@test -n "$(FILE_ID)" || (echo "Error: FILE_ID is required. Usage: make rerun FILE_ID=<id>" && exit 1)
	modal run transcribe.py --file-id $(FILE_ID) --language $(LANGUAGE) --output $(OUTPUT) --skip-whisper

# Move transcript files to ./archive
archive:
	@mkdir -p archive
	@mv -v transcript*.txt archive/ 2>/dev/null || echo "No transcript files to archive."

# Remove cached whisper results
clean:
	rm -f whisper_cache_*.json
