K=1 Chronogeometrodynamic Transformer

首个用数学证明最优结构的神经网络理论
The first neural network with mathematically proven optimal structure




K=1 Chronogeometrodynamics
https://doi.org/10.5281/zenodo.18949565

---

## ChatGPT Codex Connector

An OpenAI-powered code assistant integrated into this repository.
It wraps the OpenAI chat-completions API and exposes convenient helpers for:

- **Code generation** – produce working code from a natural-language description
- **Code completion** – fill in or extend an incomplete snippet
- **Code explanation** – plain-English description of what code does
- **Bug fixing** – identify and repair defects automatically
- **Code optimization** – improve performance or readability

### Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Run the CLI
python cli.py generate "a Python function that reverses a string"
python cli.py explain --file k1_unified.py
python cli.py fix     --file my_script.py
```

### Python SDK

```python
from codex_connector import CodexConnector

connector = CodexConnector(api_key="sk-...")

# Generate code
code = connector.generate("a function that computes Fibonacci numbers")
print(code)

# Explain existing code
explanation = connector.explain(open("k1_unified.py").read())
print(explanation)

# Async usage
import asyncio
result = asyncio.run(connector.generate_async("an async HTTP client"))
```

### CLI reference

| Command    | Description                              |
|------------|------------------------------------------|
| `generate` | Generate code from a text description    |
| `complete` | Complete an incomplete code snippet      |
| `explain`  | Explain what a piece of code does        |
| `fix`      | Identify and fix bugs                    |
| `optimize` | Optimize for performance or readability  |

Each command accepts `--file` (path), `--code` (inline string), or stdin.
Run `python cli.py <command> --help` for full option details.

### Project structure

```
codex_connector/
├── __init__.py      # Package init & public API
├── config.py        # Configuration (env vars / .env)
├── api_client.py    # OpenAI API wrapper with retry & cache
├── core.py          # High-level CodexConnector class
└── utils.py         # Helpers (logging, caching, text utils)
cli.py               # Command-line interface
examples.py          # Runnable usage examples
.env.example         # Environment variable template
requirements.txt     # Python dependencies
```

### Environment variables

| Variable              | Default              | Description                        |
|-----------------------|----------------------|------------------------------------|
| `OPENAI_API_KEY`      | *(required)*         | Your OpenAI API key                |
| `CODEX_MODEL`         | `gpt-4o`             | Model used for requests            |
| `CODEX_MAX_TOKENS`    | `2048`               | Maximum response tokens            |
| `CODEX_TEMPERATURE`   | `0.2`                | Sampling temperature (0–2)         |
| `CODEX_TIMEOUT`       | `60`                 | Request timeout in seconds         |
| `CODEX_MAX_RETRIES`   | `3`                  | Retry attempts on failure          |
| `CODEX_RETRY_DELAY`   | `1.0`                | Base delay between retries (s)     |
| `CODEX_CACHE_ENABLED` | `false`              | Enable in-memory response cache    |
| `CODEX_CACHE_TTL`     | `3600`               | Cache TTL in seconds               |
| `CODEX_LOG_LEVEL`     | `INFO`               | Logging verbosity                  |
