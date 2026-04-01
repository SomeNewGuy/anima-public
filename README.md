# ANIMA

AI platform with persistent memory, intelligent model routing, and plugin ecosystem.

ANIMA gives local language models a persistent belief system, dream synthesis, curiosity-driven exploration, and fair multi-model load balancing. It runs on your hardware, keeps your data local, and powers multiple products through a plugin architecture.

## Install

```bash
git clone https://github.com/SomeNewGuy/anima-public.git
cd anima-public
./install.sh
```

## Configure

Edit `config/settings.toml` to point at your inference server:

```toml
[inference]
inference_server = "http://localhost:8080"

[models.default]
name = "my-model"
endpoint = "http://localhost:8080"
backend = "llama"                      # llama | ollama | anthropic | openai
context_window = 16384
enabled = true
```

ANIMA works with any inference backend:
- **llama.cpp** — `llama-server` with any GGUF model
- **Ollama** — `ollama serve` with any pulled model
- **Anthropic** — Claude API (set `ANTHROPIC_API_KEY` in `.env`)
- **OpenAI** — GPT API (set `OPENAI_API_KEY` in `.env`)

## Run

```bash
./anima start        # start the platform
./anima stop         # stop
./anima status       # check what's running
```

Dashboard at [http://localhost:8900](http://localhost:8900).

## What You Get

**Memory Backbone** — Persistent beliefs extracted from conversations, documents, and exploration. Beliefs connect into a knowledge graph. Curiosity detects gaps and drives learning.

**Model Router** — Intelligent load balancing across multiple models. Capability-based scoring, health monitoring, fair queuing. One model or ten — ANIMA distributes work.

**Dream Synthesis** — Beliefs consolidate into higher-order insights during dream cycles. Cross-domain connections emerge. Requires the [ANIMA Core](https://animahub.io/core) binary (free download).

**Plugin Ecosystem** — Drop a folder in `plugins/` with a `plugin.toml` and it loads automatically. Each plugin gets isolated storage, its own dashboard, and shared inference.

## Plugins

| Plugin | Description | License |
|--------|-------------|---------|
| **Home** | Personal knowledge assistant | AGPL-3.0 |
| **DMS** | Document management + belief extraction | AGPL-3.0 |
| **Game** | AI-driven MUD world builder (Sectorfall) | AGPL-3.0 |

Additional commercial plugins available at [animahub.io](https://animahub.io).

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

## ANIMA Core (Optional)

The compiled Rust core adds dream synthesis, graph operations, and task queuing. ANIMA runs without it in memory-only mode — all features except dreams work.

Download: [animahub.io/core](https://animahub.io/core)

```bash
pip install anima-core    # when available
```

The ANIMA Core binary is provided under a separate license. See [BINARY-LICENSE](BINARY-LICENSE) for terms.

## Requirements

- Python 3.10+
- Linux or macOS (WSL2 works)
- At least one inference server (llama.cpp, Ollama, or API key)
- 8GB+ RAM recommended

## Hub

Browse and install tools, belief packs, templates, and more:

```bash
./anima hub list
./anima hub install <package>
```

Visit [animahub.io](https://animahub.io) for the full marketplace.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By submitting a pull request, you agree to the Contributor License Agreement.

## License

Copyright (c) 2026 Gerald Teeple. All rights reserved.

ANIMA Platform is licensed under [AGPL-3.0](LICENSE).

- **Use it** — run, study, modify for any purpose
- **Extend it** — build plugins, tools, integrations
- **Share back** — if you modify and serve ANIMA, share your changes

The ANIMA Core binary is provided under a [separate license](BINARY-LICENSE).
Commercial plugins and enterprise support at [animahub.io](https://animahub.io).

"ANIMA", "ANIMA Hub", and "animahub.io" are trademarks of Gerald Teeple. See [NOTICE](NOTICE).
