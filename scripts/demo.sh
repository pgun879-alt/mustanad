#!/usr/bin/env bash
# Reproducible end-to-end demo. No API key, no network, no external service.
#
# Indexes the bundled bilingual sample corpus, then asks questions in English and Arabic,
# including one that has no answer anywhere in the documents -- which the system should
# decline rather than guess at.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found. Create the environment first:" >&2
  echo "  python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

# A throwaway database, so the demo is identical every time it runs.
DEMO_DB="data/demo.sqlite3"
export MUSTANAD_DB_PATH="$DEMO_DB"
export MUSTANAD_AUTH_REQUIRED=false
export MUSTANAD_PROVIDER=extractive
export MUSTANAD_LOG_LEVEL=WARNING

rule() { printf '\n\033[1;36m%s\033[0m\n' "── $* ─────────────────────────────────────────"; }

rm -f "$DEMO_DB" "$DEMO_DB"-wal "$DEMO_DB"-shm

rule "1. Indexing the sample corpus (2 English + 2 Arabic documents)"
"$PYTHON" -m mustanad.cli ingest samples

rule "2. What is indexed"
"$PYTHON" -m mustanad.cli list

rule "3. English questions"
for question in \
  "How many days of annual leave do I get?" \
  "What is the per-diem allowance for domestic travel?" \
  "What is the first response target for a critical issue?" \
  "How long are support tickets retained after closing?"
do
  printf '\n\033[1mQ:\033[0m %s\n' "$question"
  "$PYTHON" -m mustanad.cli ask "$question" --top-k 3
done

rule "4. Arabic questions (أسئلة بالعربية)"
for question in \
  "ما هو بدل السفر الداخلي؟" \
  "ما مدة إجازة الأمومة؟" \
  "متى يكون الشحن مجانيا؟" \
  "ما مدة الضمان على الأجهزة الإلكترونية؟"
do
  printf '\n\033[1mQ:\033[0m %s\n' "$question"
  "$PYTHON" -m mustanad.cli ask "$question" --top-k 3
done

rule "5. A question the documents cannot answer"
echo "The system must decline instead of inventing an answer:"
printf '\n\033[1mQ:\033[0m %s\n' "What is the capital of Japan?"
"$PYTHON" -m mustanad.cli ask "What is the capital of Japan?"

printf '\n\033[1mQ:\033[0m %s\n' "كيف أطبخ الكسكس؟"
"$PYTHON" -m mustanad.cli ask "كيف أطبخ الكسكس؟"

rule "Done"
cat <<'EOF'
The demo index is at data/demo.sqlite3 (git-ignored).

To explore the same corpus over HTTP:

    MUSTANAD_DB_PATH=data/demo.sqlite3 MUSTANAD_AUTH_REQUIRED=false \
      ./.venv/bin/python -m mustanad.cli serve

then open http://127.0.0.1:8000 for the demo page, or
http://127.0.0.1:8000/docs for the OpenAPI console.
EOF
