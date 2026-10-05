#!/usr/bin/env bash
#
# Provision a fresh Ubuntu or Debian server to serve this API over ngrok.
#
# Installs uv, Python 3.12, Postgres, the project and its models, and ngrok,
# then runs the API and the tunnel as systemd services so both survive a logout
# and a reboot. Prints the public https url and the API key to paste into the
# Streamlit Cloud front end.
#
#   ANTHROPIC_API_KEY=sk-ant-... NGROK_AUTHTOKEN=2abc... ./scripts/provision.sh
#
# Safe to run again. Every step checks for its own result first, so a second run
# repairs a half finished one rather than duplicating it.
#
# Qdrant is deliberately not installed. It runs in process inside the API, which
# is the documented default in config/settings.example.yaml and the reason the
# API is the only process allowed to hold the collection. A container would add
# a service to operate and change nothing a reviewer can see.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Qasim-Sajjad/agentic-rag.git}"
REPO_DIR="${REPO_DIR:-$HOME/agentic-rag}"
API_PORT="${API_PORT:-8000}"
PG_DB="${PG_DB:-agentic_rag}"
PG_USER="${PG_USER:-rag}"
SERVICE_USER="${SERVICE_USER:-$USER}"
# A reserved domain keeps the url stable across restarts. Without one ngrok
# hands out a new hostname every time the tunnel starts, and the Streamlit
# secret has to be updated with it.
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
INSTALL_BROWSERS="${INSTALL_BROWSERS:-1}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
note() { printf '    %s\n' "$1"; }
die() { printf '\n\033[1;31mfailed: %s\033[0m\n' "$1" >&2; exit 1; }

require_env() {
  local name="$1" hint="$2"
  [ -n "${!name:-}" ] || die "$name is not set. $hint"
}

# ---------------------------------------------------------------- preflight

log "Checking the host"
[ "$(uname -s)" = "Linux" ] || die "this script targets Linux. Run it on the server."
command -v apt-get >/dev/null || die "no apt-get. This script targets Ubuntu or Debian."
[ "$(id -u)" -ne 0 ] || die "run as a normal user with sudo, not as root. The services run as that user."
sudo -v || die "this user needs sudo."

require_env ANTHROPIC_API_KEY "The agent and the OCR path both call Anthropic."
require_env NGROK_AUTHTOKEN "Get one from https://dashboard.ngrok.com/get-started/your-authtoken"

# Generated once and then reused, so running this script again does not
# invalidate the key already pasted into Streamlit.
RAG_API_KEY="${RAG_API_KEY:-}"
PG_PASSWORD="${PG_PASSWORD:-}"

# ------------------------------------------------------------ system packages

log "Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  ca-certificates curl git build-essential pkg-config \
  postgresql postgresql-contrib jq
note "postgres, git, build tools"

# ---------------------------------------------------------------------- uv

log "Installing uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || die "uv is still not on PATH after installing it."
note "$(uv --version)"

# -------------------------------------------------------------------- repo

log "Fetching the repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || note "pull skipped, local changes present"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
note "$REPO_DIR at $(git rev-parse --short HEAD)"

# ------------------------------------------------------------------ python

log "Installing Python 3.12 and the project"
[ -d .venv ] || uv venv --python 3.12
# The extras this API needs at runtime. `dev` is not installed: the server runs
# the service, it does not run the test suite.
uv sync --no-dev --extra fetch --extra extract --extra index --extra retrieve --extra agent
note "virtualenv at $REPO_DIR/.venv"

if [ "$INSTALL_BROWSERS" = "1" ]; then
  log "Installing browser engines for fetch tiers 2 and 3"
  note "about 1 GB. Set INSTALL_BROWSERS=0 to skip, and tiers 2 and 3 then fail"
  # The system libraries need root. The browser download must not: it lands in
  # the running user's cache, and installing it as root would put it somewhere
  # the service cannot read.
  sudo -E env "PATH=$PATH" "$(command -v uv)" run python -m playwright install-deps chromium
  uv run python -m playwright install chromium
  uv run python -m camoufox fetch
fi

# ---------------------------------------------------------------- postgres

log "Setting up Postgres"
sudo systemctl enable --now postgresql

role_exists=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'" || true)
if [ "$role_exists" != "1" ]; then
  [ -n "$PG_PASSWORD" ] || PG_PASSWORD="$(openssl rand -hex 16)"
  sudo -u postgres psql -qc "CREATE ROLE $PG_USER LOGIN PASSWORD '$PG_PASSWORD'"
  note "created role $PG_USER"
elif [ -n "$PG_PASSWORD" ]; then
  sudo -u postgres psql -qc "ALTER ROLE $PG_USER WITH PASSWORD '$PG_PASSWORD'"
  note "reset the password for $PG_USER"
else
  # The role predates this run and no password was given, so the DSN already in
  # .env is the only thing that knows it. Reusing it is correct; inventing a new
  # password here would lock the service out of its own database.
  [ -f .env ] || die "role $PG_USER exists but there is no .env holding its password. Pass PG_PASSWORD=..."
  note "reusing the existing role and the DSN already in .env"
fi

db_exists=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$PG_DB'" || true)
if [ "$db_exists" != "1" ]; then
  sudo -u postgres createdb -O "$PG_USER" "$PG_DB"
  note "created database $PG_DB"
fi

# --------------------------------------------------------------- .env file

log "Writing .env"
if [ -n "$PG_PASSWORD" ]; then
  DSN="postgresql://$PG_USER:$PG_PASSWORD@127.0.0.1:5432/$PG_DB"
else
  DSN="$(grep -m1 '^RAG__POSTGRES__DSN=' .env | cut -d= -f2-)"
  [ -n "$DSN" ] || die "no DSN in .env and no PG_PASSWORD given."
fi

umask 077
{
  echo "RAG__POSTGRES__DSN=$DSN"
  echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
  if [ -n "${SCRAPINGBEE_API_KEY:-}" ]; then
    echo "SCRAPINGBEE_API_KEY=$SCRAPINGBEE_API_KEY"
  fi
} > .env
umask 022
note "0600, holds the DSN and the Anthropic key"

# ------------------------------------------------------------ settings.yaml

log "Writing config/settings.yaml"
# Copied from the example rather than generated, so the server runs the same
# configuration as the repo documents, and then one block is replaced: the
# example ships `dev-key`, which must not be the credential on a public url.
if [ -z "$RAG_API_KEY" ]; then
  if [ -f config/settings.yaml ]; then
    RAG_API_KEY="$(uv run python -c "import yaml,sys; c=yaml.safe_load(open('config/settings.yaml')) or {}; print(next(iter((c.get('api') or {}).get('api_keys') or {}), ''))")"
  fi
  [ -n "$RAG_API_KEY" ] || RAG_API_KEY="$(openssl rand -hex 16)"
fi

RAG_API_KEY="$RAG_API_KEY" uv run python - <<'PY'
import os
import pathlib

import yaml

example = pathlib.Path("config/settings.example.yaml")
target = pathlib.Path("config/settings.yaml")
config = yaml.safe_load(example.read_text(encoding="utf-8"))

api = config.setdefault("api", {})
# Replaced, not merged. `dev-key` is in the example and in the model default,
# and a key that is in a public repository is not a key.
api["api_keys"] = {os.environ["RAG_API_KEY"]: "default"}

target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"wrote {target} with one api key")
PY

# ------------------------------------------------------- schema and sources

log "Applying migrations and seeding sources"
uv run python -m rag.db.migrate
uv run python -m rag.fetch.bootstrap
note "config/sources.yaml is a seed. The registry row is what the fetcher reads."

log "Downloading the embedding and reranking models"
note "about 300 MB, once. Without this the first question pays for it instead."
uv run python - <<'PY'
import asyncio

from rag.config.settings import get_settings
from rag.index.embed import build_embedder
from rag.retrieve.rerank import MiniLMReranker
from rag.retrieve.types import RetrievedChunk

settings = get_settings()


async def main() -> None:
    embedder = build_embedder(settings.index)
    await embedder.embed(["warm the embedding model"])
    chunk = RetrievedChunk(
        chunk_id="warmup", text="warm the reranker", score=0.0, source_url=""
    )
    await MiniLMReranker(settings.retrieve).rerank("warmup", [chunk])
    print(f"ready: {embedder.model_name}, {embedder.dims} dims")


asyncio.run(main())
PY

# ------------------------------------------------------------------- ngrok

log "Installing ngrok"
if ! command -v ngrok >/dev/null; then
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq ngrok
fi
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
NGROK_BIN="$(command -v ngrok)"
note "$($NGROK_BIN version)"

# ---------------------------------------------------------------- services

log "Installing systemd services"
UV_BIN="$(command -v uv)"
NGROK_ARGS="http $API_PORT --log stdout"
[ -n "$NGROK_DOMAIN" ] && NGROK_ARGS="http $API_PORT --domain=$NGROK_DOMAIN --log stdout"

sudo tee /etc/systemd/system/rag-api.service >/dev/null <<UNIT
[Unit]
Description=Agentic RAG API
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$REPO_DIR
# uv resolves its toolchain under HOME, and the models cache there too. systemd
# does not reliably set it, and without it the service downloads 300 MB again
# into a directory it may not be able to write.
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
# Bound to loopback on purpose. ngrok is the only way in, so the API is not
# also listening on a public interface with a static key in front of it.
ExecStart=$UV_BIN run uvicorn rag.api.main:app --host 127.0.0.1 --port $API_PORT
Restart=on-failure
RestartSec=5
# Model loading and a long PDF are both slow. A short timeout would restart the
# service in the middle of an ingest.
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/rag-ngrok.service >/dev/null <<UNIT
[Unit]
Description=ngrok tunnel for the Agentic RAG API
After=rag-api.service
Requires=rag-api.service

[Service]
User=$SERVICE_USER
# The authtoken was written to this user's ngrok config, not root's.
Environment=HOME=$HOME
ExecStart=$NGROK_BIN $NGROK_ARGS
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable rag-api.service rag-ngrok.service
sudo systemctl restart rag-api.service
sudo systemctl restart rag-ngrok.service

# ----------------------------------------------------------------- verify

log "Waiting for the API"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$API_PORT/openapi.json" >/dev/null 2>&1; then
    note "up on 127.0.0.1:$API_PORT"
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$API_PORT/openapi.json" >/dev/null \
  || die "the API did not come up. sudo journalctl -u rag-api -n 50"

log "Waiting for the tunnel"
PUBLIC_URL=""
for _ in $(seq 1 30); do
  PUBLIC_URL="$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | jq -r '.tunnels[]? | select(.proto=="https") | .public_url' | head -1)"
  [ -n "$PUBLIC_URL" ] && break
  sleep 2
done
[ -n "$PUBLIC_URL" ] || die "ngrok did not report a tunnel. sudo journalctl -u rag-ngrok -n 50"

log "Checking the public url end to end"
status=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "X-API-Key: $RAG_API_KEY" -H "ngrok-skip-browser-warning: true" \
  "$PUBLIC_URL/ingest/status")
[ "$status" = "200" ] && note "the real key is accepted" \
  || die "the public url answered $status for a valid key."

# The example config ships `dev-key`, and dict settings merge across sources,
# so this asserts it did not survive into the running process.
status=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "X-API-Key: dev-key" -H "ngrok-skip-browser-warning: true" \
  "$PUBLIC_URL/ingest/status")
[ "$status" = "401" ] && note "dev-key is rejected" \
  || die "dev-key answered $status. It is still a valid credential on a public url."

# ---------------------------------------------------------------- summary

cat <<SUMMARY

$(printf '\033[1m')Done. Put these two into the Streamlit Cloud app's secrets:$(printf '\033[0m')

    RAG_API_BASE = "$PUBLIC_URL"
    RAG_API_KEY  = "$RAG_API_KEY"

Both values are also in $REPO_DIR/config/settings.yaml and the output above.

Services:
    sudo systemctl status rag-api rag-ngrok
    sudo journalctl -u rag-api -f
    sudo systemctl restart rag-api

The url changes every time the tunnel restarts, unless you reserve a domain at
https://dashboard.ngrok.com and re-run this with NGROK_DOMAIN=your-name.ngrok-free.app

SUMMARY
