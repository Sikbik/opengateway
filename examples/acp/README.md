# ACP examples

These examples are for the experimental ACP lane on the `acp` branch.

Current scope:
- `codex` is the only real ACP adapter today
- `claude` is still a placeholder and is not ready to configure
- the ACP lane is separate from the existing HTTP / Factory gateway lane

Use these examples only after this passes on your machine:

```bash
opengateway acp doctor
```

The doctor output should show:
- `codex-ready: yes`
- `codex-issue: none`

## Start command

Every editor example in this folder launches the same stdio server:

```bash
opengateway acp serve --agent codex
```

If you want the harness to stay inside one specific repo, add `--workspace`:

```bash
opengateway acp serve --agent codex --workspace /full/path/to/repo
```

## Files

- `zed-settings.jsonc`
  - sample `settings.json` fragment for Zed custom agents
- `jetbrains-acp.json`
  - sample `~/.jetbrains/acp.json` content for JetBrains AI Assistant
- `TROUBLESHOOTING.md`
  - common failure checks and platform caveats

## Path rules

Use a real executable path in editor config.

Recommended Linux/macOS install path:

```text
~/.local/bin/opengateway
```

On Windows, point the editor at the installed Windows binary instead of a WSL path.
Do not point a Windows editor at a Linux binary inside WSL.

## Environment rule

Keep the editor, `opengateway`, and the Codex runtime in the same environment.

Good:
- Linux editor -> Linux `opengateway`
- macOS editor -> macOS `opengateway`
- Windows editor -> Windows `opengateway.exe`

Not supported as a first-class ACP setup:
- Windows editor spawning a WSL `opengateway`
- JetBrains ACP inside WSL

See `TROUBLESHOOTING.md` for the WSL caveats.
