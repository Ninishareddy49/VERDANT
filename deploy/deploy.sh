#!/usr/bin/env bash
# One-shot VERDANT deploy: ML backend -> Hugging Face Space, frontend -> Vercel.
#
#   deploy/deploy.sh            # deploy both
#   deploy/deploy.sh backend    # backend only
#   deploy/deploy.sh frontend   # frontend only
#
# Accounts: tokens in .env.deploy win; otherwise the global CLI logins (deploy/login.sh)
# Overrides: SPACE_NAME (default verdant-api), VERCEL_PROJECT (default verdant)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_NAME="${SPACE_NAME:-verdant-api}"
VERCEL_PROJECT="${VERCEL_PROJECT:-verdant}"
TARGET="${1:-all}"

# Per-project accounts (see .env.deploy); exported so every CLI picks them up
[[ -f "$ROOT/.env.deploy" ]] && { set -a; source "$ROOT/.env.deploy"; set +a; }
VERCEL_ARGS=(); [[ -n "${VERCEL_TOKEN:-}" ]] && VERCEL_ARGS=(--token "$VERCEL_TOKEN")

HF_USER="$(hf auth whoami --format quiet 2>/dev/null | head -1)" || true
[[ -n "$HF_USER" ]] || { echo "Not logged in to Hugging Face — run deploy/login.sh"; exit 1; }
SPACE_ID="$HF_USER/$SPACE_NAME"
# Space hostnames are lowercase with non-alphanumerics turned into hyphens
SPACE_HOST="$(echo "$HF_USER-$SPACE_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '-').hf.space"

deploy_backend() {
  echo "==> Backend -> https://huggingface.co/spaces/$SPACE_ID"
  hf repos create "$SPACE_ID" --type space --sdk docker --public --exist-ok >/dev/null

  # Stage only what the image needs, plus the Space card
  local stage; stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' RETURN
  cp -r "$ROOT/backend/app" "$ROOT/backend/Dockerfile" "$ROOT/backend/.dockerignore" \
        "$ROOT/backend/requirements-deploy.txt" "$ROOT/backend/download_models.py" "$stage/"
  cp "$ROOT/deploy/hf-space-README.md" "$stage/README.md"
  find "$stage" -name __pycache__ -prune -exec rm -rf {} +

  # Secrets come from backend/.env (never uploaded as a file)
  if [[ -f "$ROOT/backend/.env" ]]; then
    grep -E '^(OLLAMA_[A-Z_]+)=' "$ROOT/backend/.env" | hf spaces secrets add "$SPACE_ID" --secrets-file - >/dev/null
    echo "    synced OLLAMA_* secrets from backend/.env"
  fi

  hf upload "$SPACE_ID" "$stage" . --repo-type space --delete '*' \
    --commit-message "deploy $(git -C "$ROOT" rev-parse --short HEAD)" >/dev/null
  echo "    uploaded; waiting for build (first build ~10 min)..."
  hf spaces wait "$SPACE_ID" --timeout 30m
  curl -fsS "https://$SPACE_HOST/api/health" && echo
}

deploy_frontend() {
  echo "==> Frontend -> Vercel (/api proxied to https://$SPACE_HOST)"
  cat > "$ROOT/frontend/vercel.json" <<EOF
{
  "\$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "vite",
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://$SPACE_HOST/api/:path*" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
EOF
  cd "$ROOT/frontend"
  [[ -f .vercel/project.json ]] || vercel link --yes --project "$VERCEL_PROJECT" "${VERCEL_ARGS[@]}"
  vercel deploy --prod --yes "${VERCEL_ARGS[@]}"
}

case "$TARGET" in
  backend)  deploy_backend ;;
  frontend) deploy_frontend ;;
  all)      deploy_backend; deploy_frontend ;;
  *) echo "usage: $0 [all|backend|frontend]"; exit 2 ;;
esac
