#!/usr/bin/env bash
# One-shot VERDANT deploy: ML backend -> Render (Docker, free plan), frontend -> Vercel.
#
#   deploy/deploy.sh            # deploy both
#   deploy/deploy.sh backend    # backend only
#   deploy/deploy.sh frontend   # frontend only
#
# Account: this project lives on Ninisha's accounts — `acct run ninisha -- deploy/deploy.sh`.
# Render builds from GitHub, so the script pushes $BRANCH to origin first.
# Overrides: SERVICE_NAME (verdant-api), VERCEL_PROJECT (verdant), BRANCH (main), REGION (singapore)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-verdant-api}"
VERCEL_PROJECT="${VERCEL_PROJECT:-verdant}"
BRANCH="${BRANCH:-main}"
REGION="${REGION:-singapore}"
TARGET="${1:-all}"

VERCEL_ARGS=(); [[ -n "${VERCEL_TOKEN:-}" ]] && VERCEL_ARGS=(--token "$VERCEL_TOKEN")
RENDER_API="https://api.render.com/v1"

render_api() {  # METHOD PATH [JSON]
  curl -fsS -X "$1" "$RENDER_API$2" -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json" ${3:+-d "$3"}
}

# Prints "ID URL" of the backend service, or nothing if it doesn't exist yet
find_service() {
  render services -o json | python3 -c '
import sys, json
for s in json.load(sys.stdin):
    s = s.get("service", s)
    if s.get("name") == sys.argv[1]:
        print(s["id"], s["serviceDetails"]["url"])' "$SERVICE_NAME"
}

# OLLAMA_* lines from backend/.env (gitignored) become Render env vars
backend_env() { [[ -f "$ROOT/backend/.env" ]] && grep -E '^OLLAMA_[A-Z_]+=' "$ROOT/backend/.env" || true; }

wait_for_deploy() {  # SERVICE_ID
  echo "    building (first build ~5-10 min)..."
  local status
  while :; do
    status="$(render_api GET "/services/$1/deploys?limit=1" | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["deploy"]["status"])')"
    case "$status" in
      live) break ;;
      build_failed|update_failed|canceled|deactivated|pre_deploy_failed)
        echo "    deploy $status — logs: render logs -r $1 -o text"; exit 1 ;;
    esac
    sleep 15
  done
}

deploy_backend() {
  echo "==> Backend -> Render ($SERVICE_NAME, branch $BRANCH)"
  git -C "$ROOT" push -q origin "$BRANCH"
  local repo; repo="$(git -C "$ROOT" remote get-url origin)"

  read -r SERVICE_ID SERVICE_URL <<<"$(find_service)" || true
  if [[ -z "${SERVICE_ID:-}" ]]; then
    local env_args=()
    while IFS= read -r kv; do [[ -n "$kv" ]] && env_args+=(--env-var "$kv"); done <<<"$(backend_env)"
    render services create --confirm -o json --name "$SERVICE_NAME" --type web_service \
      --runtime docker --plan free --region "$REGION" --repo "${repo%.git}" --branch "$BRANCH" \
      --root-directory backend --health-check-path /api/health --auto-deploy=false \
      "${env_args[@]}" >/dev/null
    read -r SERVICE_ID SERVICE_URL <<<"$(find_service)"
    echo "    created $SERVICE_ID"
  else
    while IFS='=' read -r k v; do
      [[ -n "$k" ]] && render_api PUT "/services/$SERVICE_ID/env-vars/$k" "{\"value\":\"$v\"}" >/dev/null
    done <<<"$(backend_env)"
    render deploys create "$SERVICE_ID" --commit "$(git -C "$ROOT" rev-parse "$BRANCH")" --confirm -o json >/dev/null
  fi

  wait_for_deploy "$SERVICE_ID"
  curl -fsS "$SERVICE_URL/api/health" && echo
}

deploy_frontend() {
  read -r _ SERVICE_URL <<<"$(find_service)" || true
  [[ -n "${SERVICE_URL:-}" ]] || { echo "Backend service '$SERVICE_NAME' not found — deploy the backend first"; exit 1; }
  echo "==> Frontend -> Vercel (/api proxied to $SERVICE_URL)"
  cat > "$ROOT/frontend/vercel.json" <<EOF
{
  "\$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "vite",
  "rewrites": [
    { "source": "/api/:path*", "destination": "$SERVICE_URL/api/:path*" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
EOF
  cd "$ROOT/frontend"
  # Re-link when deploying under a different account than the last link
  if [[ ! -f .vercel/project.json || "$(cat .vercel/acct 2>/dev/null)" != "${ACCT:-global}" ]]; then
    rm -rf .vercel
    vercel link --yes --project "$VERCEL_PROJECT" "${VERCEL_ARGS[@]}"
    echo "${ACCT:-global}" > .vercel/acct
  fi
  vercel deploy --prod --yes "${VERCEL_ARGS[@]}"
}

case "$TARGET" in
  backend)  deploy_backend ;;
  frontend) deploy_frontend ;;
  all)      deploy_backend; deploy_frontend ;;
  *) echo "usage: $0 [all|backend|frontend]"; exit 2 ;;
esac
