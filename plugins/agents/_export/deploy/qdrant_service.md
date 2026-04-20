# Qdrant deployment

Agent graphs that use `MemoryStoreNode` with the default `qdrant` backend
run Qdrant **in-process** via `qdrant-client` local mode — no separate
server. The exported script just calls `QdrantClient(path="./qdrant_data")`
and everything works.

If you outgrow the local path mode (multiple processes accessing the same
store, cross-host access, or the data directory gets large), switch to a
standalone Qdrant server and change `path=` to a URL.

## Docker — the fastest path

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest
```

Then update your graph's `MemoryStoreNode.path` from
`./qdrant_data` to `http://localhost:6333`. No code change in the node —
`qdrant-client` auto-detects the URL vs. path.

## Linux — systemd

Install the Qdrant binary (see https://qdrant.tech/documentation/), then:

```ini
# /etc/systemd/system/qdrant.service
[Unit]
Description=Qdrant vector database
After=network-online.target

[Service]
Type=exec
ExecStart=/usr/local/bin/qdrant
WorkingDirectory=/var/lib/qdrant
User=qdrant
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qdrant
```

## Windows — nssm

```cmd
nssm install qdrant "C:\Program Files\qdrant\qdrant.exe"
nssm set qdrant AppDirectory "C:\Program Files\qdrant"
nssm start qdrant
```

## Migrating existing local data

Local-mode data lives under `./qdrant_data` (or whatever `path=` you set).
The on-disk format is identical to a standalone Qdrant server's storage.
Copy the directory into your server's `storage/` and it'll mount.
