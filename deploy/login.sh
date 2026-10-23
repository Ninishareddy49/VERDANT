#!/usr/bin/env bash
# Log every deploy CLI in once. Each one opens a browser or prints a code.
# Already-authenticated tools are skipped, so this is safe to re-run.
set -uo pipefail

step() { printf '\n\033[1;32m==> %s\033[0m\n' "$1"; }

step "GitHub (gh)"
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https
gh auth setup-git >/dev/null 2>&1

step "Hugging Face (hf) — backend hosting"
echo "Create a WRITE token at https://huggingface.co/settings/tokens and paste it below."
hf auth whoami >/dev/null 2>&1 || hf auth login

step "Vercel — frontend hosting"
vercel whoami >/dev/null 2>&1 || vercel login

step "Supabase — database/auth (for later)"
supabase projects list >/dev/null 2>&1 || supabase login

step "Render — alternative hosting (for later)"
render whoami >/dev/null 2>&1 || render login

step "Railway — alternative hosting (for later)"
railway whoami >/dev/null 2>&1 || railway login

step "Status"
printf '%-10s %s\n' gh "$(gh api user -q .login 2>/dev/null || echo '✗')"
printf '%-10s %s\n' hf "$(hf auth whoami --format quiet 2>/dev/null | head -1 || echo '✗')"
printf '%-10s %s\n' vercel "$(vercel whoami 2>/dev/null || echo '✗')"
printf '%-10s %s\n' supabase "$(supabase projects list >/dev/null 2>&1 && echo ok || echo '✗')"
printf '%-10s %s\n' render "$(render whoami -o text 2>/dev/null | head -1 || echo '✗')"
printf '%-10s %s\n' railway "$(railway whoami 2>/dev/null || echo '✗')"
