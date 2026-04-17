// WebSocket JSON-RPC 2.0 client for the NodeTool server.
// The server lives at ws://127.0.0.1:9800 (configurable). Every request gets a
// monotonic integer id; the server echoes it back so we can match responses to
// pending promises. On disconnect, all pending promises reject.

export type RpcParams = Record<string, unknown>;

type Pending = {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
};

export type ConnState = "idle" | "connecting" | "open" | "closed";

export class RpcClient {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private url: string;
  private onState: (s: ConnState) => void;

  constructor(url: string, onState: (s: ConnState) => void) {
    this.url = url;
    this.onState = onState;
  }

  connect(): void {
    this.onState("connecting");
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onopen = () => this.onState("open");
    ws.onclose = () => {
      this.onState("closed");
      // Reject all in-flight
      for (const p of this.pending.values()) p.reject(new Error("Disconnected"));
      this.pending.clear();
    };
    ws.onerror = () => {
      // onclose fires after this, cleanup happens there
    };
    ws.onmessage = (ev) => this.handleMessage(ev.data);
  }

  private handleMessage(raw: string): void {
    let msg: { id?: number; result?: unknown; error?: { message?: string } };
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    if (msg.id === undefined) return; // notification, unused for now
    const pending = this.pending.get(msg.id);
    if (!pending) return;
    this.pending.delete(msg.id);
    if (msg.error) {
      pending.reject(new Error(msg.error.message || "RPC error"));
    } else {
      pending.resolve(msg.result);
    }
  }

  call<T = unknown>(method: string, params: RpcParams = {}): Promise<T> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Not connected"));
    }
    const id = this.nextId++;
    const frame = { jsonrpc: "2.0", method, params, id };
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.ws!.send(JSON.stringify(frame));
    });
  }
}
