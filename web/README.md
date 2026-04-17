# NodeTool Web Frontend

React + TypeScript + Vite. Connects to the Python backend (`server.py`) over
WebSocket JSON-RPC at `ws://127.0.0.1:9800`.

## Setup

```bash
cd web
npm install
```

## Development

Start the backend in one terminal:

```bash
python server.py
```

Start the dev server in another:

```bash
cd web
npm run dev
```

Open http://localhost:5173.

## Status

- [x] WebSocket RPC client with auto-reconnect hooks
- [x] Palette (search + categorized node list from `get_registry`)
- [x] Connection status badge
- [ ] Graph canvas (React Flow) — next
- [ ] Inspector panel
- [ ] Bottom tab panels (Output / Code / Training / Robotics)
- [ ] Plugin panel builders (discovered via `get_plugin_panels`)
