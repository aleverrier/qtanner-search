#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat <<'EOS' > "${TMP_DIR}/qtanner"
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec python -m qtanner.cli "$@"
EOS
chmod +x "${TMP_DIR}/qtanner"
export PATH="${TMP_DIR}:${PATH}"

qtanner --help >/dev/null
qtanner search --help >/dev/null
qtanner refine --help >/dev/null
qtanner publish --help >/dev/null

echo "[smoke] qtanner CLI help checks passed."
