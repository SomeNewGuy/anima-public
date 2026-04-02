# ANIMA

AI platform with persistent memory, intelligent model routing, and plugin ecosystem.

ANIMA gives local language models a persistent belief system, dream synthesis, curiosity-driven exploration, and fair multi-model load balancing. It runs on your hardware, keeps your data local, and powers multiple products through a plugin architecture.

**ANIMA requires the compiled ANIMA Core engine, which is installed automatically by the installer.**

## Install

```bash
git clone https://github.com/SomeNewGuy/anima-public.git
cd anima-public
./install.sh   # installs dependencies + ANIMA Core binary
```

Verify it worked:

```bash
./anima status
```

First startup may take a few seconds while embedding models initialize.

## Configure

Edit `config/settings.toml` to add your inference server:

```toml
[models.default]
name = "my-model"
endpoint = "http://localhost:8080"
backend = "llama"                      # llama | ollama | anthropic
context_window = 16384
enabled = true
```

ANIMA works with any inference backend:
- **llama** — `llama-server` with any GGUF model
- **ollama** — `ollama serve` with any pulled model
- **anthropic** — Claude API (set `ANTHROPIC_API_KEY` in `.env`)
- Any **OpenAI-compatible** endpoint (vLLM, TabbyAPI, etc.)

## Run

```bash
./anima start        # start the platform
./anima stop         # stop
./anima status       # check what's running
```

Dashboard at [http://localhost:8900](http://localhost:8900).

## What You Get

**Memory Backbone** — Persistent beliefs extracted from conversations, documents, and exploration. Beliefs connect into a knowledge graph. Curiosity identifies gaps and triggers new belief generation.

**Model Router** — Intelligent load balancing across multiple models. Capability-based scoring, health monitoring, fair queuing. One model or ten — ANIMA distributes work.

**Dream Synthesis** — Beliefs consolidate into higher-order insights during dream cycles. Cross-domain connections emerge. Powered by the ANIMA Core compiled engine (installed automatically).

**Plugin Ecosystem** — Drop a folder in `plugins/` with a `plugin.toml` and it loads automatically. Each plugin gets isolated storage, its own dashboard, and shared inference.

## Plugins

ANIMA ships with a plugin template to get you started. Additional plugins available at [animahub.io](https://animahub.io) — including research, document management, game world building, and more.

## Hub

Browse and install tools, belief packs, templates, and more:

```bash
./anima hub list
./anima hub install <package>
```

Visit [animahub.io](https://animahub.io) for the full marketplace (coming soon).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  ANIMA Core                      │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Router   │  │ Embeddings│  │  Tool Registry │  │
│  │ (shared)  │  │ (shared)  │  │   (shared)    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         Plugin Instance (per plugin)      │   │
│  │  Own DB  │  Own Memory  │  Own Dashboard  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

- **One port** (8900), all plugins load automatically
- **Plugins control themselves** — start/stop from their dashboards
- **Data isolation** — each plugin gets its own SQLite + ChromaDB
- **Shared inference** — all plugins use the same model pool

## ANIMA Core (Required)

The open-source platform is the orchestration layer. Core intelligence runs in the compiled ANIMA Core engine — beliefs, dream synthesis, model routing, graph operations, and task queuing.

The `install.sh` script downloads and installs the correct binary for your platform automatically.

### Platform Support

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux | x86_64 | Available |
| Linux | aarch64 | Coming soon |
| macOS | Apple Silicon (arm64) | Coming soon |
| macOS | Intel (x86_64) | Coming soon |
| Windows | x86_64 | Coming soon |

## Requirements

- Python 3.10+
- Linux, macOS, or Windows (WSL2 works)
- At least one inference server (llama.cpp, Ollama, or API key)
- 8GB+ RAM recommended

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By submitting a pull request, you agree to the Contributor License Agreement.

## License

Copyright (c) 2026 Gerald Teeple. All rights reserved.

ANIMA Platform is licensed under [AGPL-3.0](LICENSE).

- **Use it** — run, study, modify for any purpose
- **Extend it** — build plugins, tools, integrations
- **Share back** — if you modify and serve ANIMA, share your changes

The ANIMA Core binary is **not** covered by the AGPL license. It is provided under a [separate license](BINARY-LICENSE).

Commercial plugins and enterprise support at [animahub.io](https://animahub.io).

"ANIMA", "ANIMA Hub", and "animahub.io" are trademarks of Gerald Teeple. See [NOTICE](NOTICE).
