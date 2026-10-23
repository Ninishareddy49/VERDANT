#!/usr/bin/env bash
# Show which account each CLI acts as from this project (project token vs global login).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/.env.deploy" ]] && { set -a; source "$ROOT/.env.deploy"; set +a; }

src() { [[ -n "${!1:-}" ]] && echo "project" || echo "global"; }
VT=(); [[ -n "${VERCEL_TOKEN:-}" ]] && VT=(--token "$VERCEL_TOKEN")

printf '%-9s %-8s %s\n' CLI SOURCE ACCOUNT
printf '%-9s %-8s %s\n' gh       "$(src GH_TOKEN)"              "$(gh api user -q .login 2>/dev/null || echo '✗ not logged in')"
printf '%-9s %-8s %s\n' hf       "$(src HF_TOKEN)"              "$(u=$(hf auth whoami --format quiet 2>/dev/null | head -1); echo "${u:-✗ not logged in}")"
printf '%-9s %-8s %s\n' vercel   "$(src VERCEL_TOKEN)"          "$(vercel whoami "${VT[@]}" 2>/dev/null || echo '✗ not logged in')"
printf '%-9s %-8s %s\n' supabase "$(src SUPABASE_ACCESS_TOKEN)" "$(supabase orgs list -o json 2>/dev/null | python3 -c 'import sys,json;print(", ".join(o["name"] for o in json.load(sys.stdin)))' 2>/dev/null || echo '✗ not logged in')"
printf '%-9s %-8s %s\n' render   "$(src RENDER_API_KEY)"        "$(render whoami -o json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["email"])' 2>/dev/null || echo '✗ not logged in')"
