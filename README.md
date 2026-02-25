# opengateway

`opengateway` is a local OAuth gateway for OpenAI-compatible workflows. It runs on your machine, handles ChatGPT OAuth login, and exposes OpenAI-style endpoints your tooling can call.

## Current support

- Fully supported today: **Factory Droid**
- Planned next: additional harness integrations beyond Factory Droid

## What it does

- Local proxy on `127.0.0.1:42069` (default)
- Browser and headless OAuth login
- OpenAI-compatible endpoints:
  - `GET /healthz`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/responses`

## Install

```bash
git clone https://github.com/Sikbik/opengateway && cd opengateway
./bin/install
```

After install, use `opengateway` directly from your shell.

## Quick start (Factory Droid)

```bash
opengateway setup
```

This runs init + start + login + Factory config update.
It merges models into `~/.factory/config.json` and writes a timestamped backup when that file already exists.

Headless login option:

```bash
opengateway setup --headless
```

Browser login opens manually by default (URL is printed):

```bash
opengateway setup --open-browser
```

## Verify your setup

```bash
opengateway status
opengateway self-test
```

Logs:

```bash
opengateway logs -f
```

## Use with Factory Droid

1. Restart `droid` if it is already running.
2. Open the model picker (`/model`).
3. Select one of the custom `GPT-*` entries added by `opengateway setup`.

When you choose that model, requests route through `opengateway` to OpenAI.

## Useful commands

```bash
opengateway start
opengateway stop
opengateway status
opengateway login
opengateway login --open-browser
opengateway login headless
opengateway self-test
opengateway show-key
opengateway factory-config
opengateway doctor
opengateway logs -f
```

## Runtime paths

Defaults:

- Config: `~/.config/opengateway/config.yaml`
- Data/Binary: `~/.local/share/opengateway`
- State/Logs: `~/.local/state/opengateway`
- Auth files: `~/.config/opengateway/auth`

Overrides:

- `OPENGATEWAY_CONFIG_DIR`
- `OPENGATEWAY_DATA_DIR`
- `OPENGATEWAY_STATE_DIR`
- `OPENGATEWAY_AUTH_DIR`
- Plus standard `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`

## Login note

- By default, browser login prints the OAuth URL and waits for you to open it.
- Use `--open-browser` if you want `opengateway` to try opening the URL automatically.
