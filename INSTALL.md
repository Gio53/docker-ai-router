# Docker AI Router — server install

OpenAI-compatible local proxy. Classifies each chat request with Laya, then
forwards simple tasks to FreeLLMAPI and complex tasks to Z.ai.

Listens on `127.0.0.1:8081` by default (avoids Nextcloud on 8080).

## What the server assistant must do

1. Clone or pull this repo onto the server.
2. Create `/etc/docker-ai-router.env` from the example below. **Do not commit
   real API keys.** Ask the human for `FREE_LLM_API_KEY` and `ZAI_API_KEY` if
   they are not already on the machine.
3. Install into `/opt/docker-ai-router`, create a venv, install Python deps.
4. Install the systemd unit and enable the service.
5. Confirm `curl http://127.0.0.1:8081/health` returns `{"status":"ok"}`.

First start may take several minutes while Laya downloads (~800 MB) and loads.

## Install commands (run on the server)

Assume this repo is cloned to `~/docker-ai-router` (adjust if the clone path differs).

```bash
SRC=~/docker-ai-router

sudo mkdir -p /opt/docker-ai-router
sudo cp "$SRC/proxy.py" "$SRC/requirements.txt" /opt/docker-ai-router/

# Secrets: create env file if missing. Prefer copying a filled file the human
# provides. Otherwise start from the example and fill REPLACE_ME values.
if [ ! -f /etc/docker-ai-router.env ]; then
  sudo cp "$SRC/docker-ai-router.env.example" /etc/docker-ai-router.env
  echo "EDIT /etc/docker-ai-router.env and set FREE_LLM_API_KEY and ZAI_API_KEY"
fi
sudo chmod 600 /etc/docker-ai-router.env

sudo python3 -m venv /opt/docker-ai-router/.venv
sudo /opt/docker-ai-router/.venv/bin/pip install -U pip
sudo /opt/docker-ai-router/.venv/bin/pip install -r /opt/docker-ai-router/requirements.txt

sudo cp "$SRC/docker-ai-router.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now docker-ai-router
sudo systemctl status docker-ai-router --no-pager
```

## Verify

```bash
curl -s http://127.0.0.1:8081/health
sudo journalctl -u docker-ai-router -n 50 --no-pager
```

## Client usage

Point any OpenAI-compatible client at:

- Base URL: `http://127.0.0.1:8081/v1`
- Or from another machine on the LAN only after setting `LISTEN_HOST=0.0.0.0`
  in `/etc/docker-ai-router.env` and restarting the service.

No inbound API key is required; this is a local proxy.

## Required env vars

| Variable | Required | Notes |
|---|---|---|
| `FREE_LLM_API_KEY` | yes | FreeLLMAPI key |
| `FREE_LLM_BASE_URL` | yes | e.g. `http://192.168.1.204:3002/v1` |
| `ZAI_API_KEY` | yes | Z.ai API key |
| `ZAI_BASE_URL` | no | default `https://api.z.ai/api/coding/paas/v4` (Coding Plan). General pay-as-you-go is `https://api.z.ai/api/paas/v4` |
| `ZAI_MODEL` | no | default `glm-5.3-flash` (Coding Plan). `glm-5.2` is auto-routed to `glm-5.3` on Coding Plan |
| `ZAI_THINKING` | no | default `disabled` when the model allows it; GLM-5.3 family forces thinking |
| `ZAI_REASONING_EFFORT` | no | default `low` (faster). Use `medium`/`high`/`max` for harder tasks |
| `ZAI_MAX_TOKENS` | no | default `4096` when the client omits a limit |
| `ROUTER_COMPLEX_SCORE_AT` | no | default `2.0` — Laya difficulty at/above this goes to Z.ai |
| `ROUTER_COMPLEX_TOOLS_AT` | no | default `0.5` — Laya needs_tools at/above this goes to Z.ai |
| `ROUTER_FORCE_COMPLEX_KEYWORDS` | no | comma list forced to Z.ai; unset = Pelican/Wings/Immich/Nextcloud list; empty = disable |
| `LISTEN_HOST` | no | default `127.0.0.1` |
| `LISTEN_PORT` | no | default `8081` (do not use 8080; Nextcloud uses it) |

## Update after a git pull

```bash
SRC=~/docker-ai-router
cd "$SRC" && git pull
sudo cp "$SRC/proxy.py" "$SRC/requirements.txt" /opt/docker-ai-router/
sudo /opt/docker-ai-router/.venv/bin/pip install -r /opt/docker-ai-router/requirements.txt
sudo cp "$SRC/docker-ai-router.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart docker-ai-router
```
