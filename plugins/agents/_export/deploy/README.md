# Deploying an exported agent graph

Agent graphs exported from node-tool are plain Python scripts. To run one
on another machine you need three things:

  1. **Python 3.10+** — the script uses `typing`-style type hints only.
  2. **The `requirements.txt`** that was generated alongside the script —
     run `pip install -r requirements.txt`. Per the dependency-minimality
     invariant (§G.3 of the plugin design), a pure-LLM graph's
     requirements omit torch / sentence-transformers / qdrant-client.
  3. **Any external backends the graph touches** — usually an Ollama
     daemon, optionally a remote Qdrant.

## Backend guides

- [ollama_service.md](./ollama_service.md) — Windows `nssm` + Linux
  `systemd` recipes, env-var conventions.
- [qdrant_service.md](./qdrant_service.md) — Docker + systemd + nssm for
  running Qdrant as a server; migrating local-mode data to a server.

## Running the script

```bash
pip install -r requirements.txt
python my_agent_graph.py
```

If the script calls Ollama and no models are pulled, run
`ollama pull <model-name>` before launching. If it uses MemoryStore, the
on-disk directory is created on first call.

## Persisting a long-running agent

Standalone agent scripts exit after one turn. To host one as a service that
takes stdin / HTTP requests, wrap `main()` in your preferred service
runner. The exported code is plain Python with no framework lock-in — the
same functions work inside FastAPI, Flask, a Discord bot, etc.

## No Docker emission in v1

The plugin intentionally does NOT emit a `Dockerfile`. The shapes of
useful Dockerfiles diverge too much across deployments — GPU requirements,
base image choices, CI-vs-prod builds. A minimal hand-written Dockerfile
on top of the generated `requirements.txt` is three lines:

```dockerfile
FROM python:3.12-slim
COPY requirements.txt my_agent_graph.py ./
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "my_agent_graph.py"]
```

Add GPU bases / volume mounts / Ollama sidecar as needed.
