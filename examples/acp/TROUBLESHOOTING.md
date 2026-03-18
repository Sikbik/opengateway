# ACP troubleshooting

This guide is for the experimental ACP lane on the `acp` branch.

## 1. Verify the local runtime first

Before touching editor config, make sure the local runtime is actually ready:

```bash
opengateway acp doctor
```

Expected signals:
- `codex-ready: yes`
- `codex-issue: none`

If `codex-ready: no`, fix that first.

## 2. Run the server manually

If the editor says the agent failed to start, run the ACP server directly:

```bash
opengateway acp serve --agent codex
```

You should see the experimental banner on stderr and then the process should wait for stdio input.

If it exits immediately, the problem is local runtime setup, not the editor.

## 3. Check the Codex runtime directly

If ACP doctor is not enough, verify Codex itself:

```bash
codex --version
codex exec --help
```

`opengateway` currently expects `codex exec` to support:
- `--json`
- `--ephemeral`
- `--skip-git-repo-check`
- `-C` or `--cd`

## 4. Use full executable paths in editor config

Do not rely on editor-specific PATH inheritance if the agent fails to spawn.

Prefer:
- Linux/macOS: `~/.local/bin/opengateway`
- Windows: full installed `opengateway.exe` path

JetBrains especially should use a full path in `acp.json`.

## 5. Check ACP logs and journals

Useful commands:

```bash
opengateway acp sessions
opengateway acp inspect <session-id>
```

Common things to look for:
- workspace path invalid
- runtime missing
- protocol mismatch
- session crashed

## 6. Zed-specific notes

Per Zed's ACP docs:
- custom agents are configured under `agent_servers`
- `dev: open acp logs` shows the ACP traffic between Zed and the agent

If the agent does not appear:
- check `settings.json` formatting
- make sure the `command` path is valid
- restart Zed if needed after changing settings

## 7. JetBrains-specific notes

Per JetBrains AI Assistant docs:
- custom agents live in `~/.jetbrains/acp.json`
- use `Add Custom Agent` or edit that file directly
- if the agent is not shown, check JSON formatting and restart the IDE

If the agent fails to start:
- use the full executable path
- run the same command manually in a terminal
- collect IDE ACP logs before filing a bug

## 8. Windows and WSL caveats

These are the important environment boundaries for this ACP MVP:

- keep the editor and `opengateway` in the same environment
- do not point a Windows editor at a WSL `opengateway` binary
- do not point a Linux/WSL editor at a Windows `opengateway.exe`

Important JetBrains caveat:
- JetBrains documents ACP agents, but WSL is not a first-class ACP target in this setup
- for JetBrains on Windows, use a Windows-native `opengateway.exe`

Important Windows caveat:
- if your project and Codex runtime live only in WSL, the current ACP MVP is not the right cross-boundary setup
- the HTTP/Factory lane and the desktop control app have separate Windows+WSL behavior; that does not mean ACP should cross that boundary

## 9. Workspace pinning

If the editor should always run the agent inside one repo, add `--workspace`:

```json
{
  "args": [
    "acp",
    "serve",
    "--agent",
    "codex",
    "--workspace",
    "/full/path/to/repo"
  ]
}
```

The path must exist and must be a directory.
