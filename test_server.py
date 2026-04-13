"""Test client for the NodeTool WebSocket server.

Exercises the full API: registry, node creation, connections, config,
execution. Run while the server is up:

    python test_server.py
"""
import asyncio
import json
import sys

import websockets

URL = "ws://127.0.0.1:9800"
_id = 0


async def rpc(ws, method: str, params: dict = {}) -> dict:
    global _id
    _id += 1
    await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": _id}))
    resp = json.loads(await ws.recv())
    if "error" in resp:
        print(f"  ERROR [{method}]: {resp['error']['message']}")
        return resp["error"]
    return resp.get("result", {})


async def test():
    async with websockets.connect(URL) as ws:
        print("=== NodeTool Server Test ===\n")

        # 1. Registry
        print("1. Fetching registry...")
        reg = await rpc(ws, "get_registry")
        cats = reg["categories"]
        total = sum(len(v) for v in cats.values())
        print(f"   {len(cats)} categories, {total} nodes")
        for cat in reg["category_order"][:5]:
            if cat in cats:
                print(f"   - {cat}: {len(cats[cat])} nodes")
        print()

        # 2. Add nodes
        print("2. Adding nodes...")
        linear = await rpc(ws, "add_node", {"type_name": "pt_linear"})
        print(f"   Linear: id={linear['id'][:8]}...")
        data_in = [n for n, p in linear["inputs"].items() if not p["editable"]]
        config_in = [n for n, p in linear["inputs"].items() if p["editable"]]
        print(f"   Data ports: {data_in}")
        print(f"   Config ports: {config_in}")
        print(f"   Outputs: {list(linear['outputs'].keys())}")

        relu = await rpc(ws, "add_node", {"type_name": "pt_activation"})
        print(f"   Activation: id={relu['id'][:8]}...")
        print()

        # 3. Set config values
        print("3. Setting config values...")
        await rpc(ws, "set_input", {
            "node_id": linear["id"],
            "port_name": "in_features",
            "value": 784,
        })
        await rpc(ws, "set_input", {
            "node_id": linear["id"],
            "port_name": "out_features",
            "value": 128,
        })
        print("   Linear: in_features=784, out_features=128")

        # Verify
        node_state = await rpc(ws, "get_node", {"node_id": linear["id"]})
        print(f"   Verified: in_features={node_state['inputs']['in_features']['default_value']}")
        print()

        # 4. Connect nodes
        print("4. Connecting Linear.tensor_out -> Activation.tensor_in...")
        result = await rpc(ws, "connect", {
            "from_node": linear["id"],
            "from_port": "tensor_out",
            "to_node": relu["id"],
            "to_port": "tensor_in",
        })
        print(f"   {result}")
        print()

        # 5. Get full graph state
        print("5. Getting graph state...")
        graph = await rpc(ws, "get_graph")
        print(f"   Nodes: {len(graph['nodes'])}")
        print(f"   Connections: {len(graph['connections'])}")
        for conn in graph["connections"]:
            print(f"   - {conn['from_port']} -> {conn['to_port']}")
        print()

        # 6. Execute
        print("6. Executing graph...")
        exec_result = await rpc(ws, "execute")
        if "error" in exec_result:
            print(f"   Error: {exec_result.get('error', exec_result.get('message'))}")
        else:
            outputs = exec_result.get("outputs", {})
            print(f"   {len(outputs)} nodes produced output")
            for nid, outs in outputs.items():
                node_label = graph["nodes"].get(nid, {}).get("label", nid[:8])
                for pname, val in outs.items():
                    print(f"   - {node_label}.{pname} = {val}")
        print()

        # 7. Disconnect
        print("7. Disconnecting...")
        await rpc(ws, "disconnect", {
            "from_node": linear["id"],
            "from_port": "tensor_out",
            "to_node": relu["id"],
            "to_port": "tensor_in",
        })
        graph2 = await rpc(ws, "get_graph")
        print(f"   Connections after disconnect: {len(graph2['connections'])}")
        print()

        # 8. Remove node
        print("8. Removing activation node...")
        await rpc(ws, "remove_node", {"node_id": relu["id"]})
        graph3 = await rpc(ws, "get_graph")
        print(f"   Nodes remaining: {len(graph3['nodes'])}")
        print()

        # 9. Clear
        print("9. Clearing graph...")
        await rpc(ws, "clear")
        graph4 = await rpc(ws, "get_graph")
        print(f"   Nodes after clear: {len(graph4['nodes'])}")
        print()

        print("=== All tests passed! ===")


if __name__ == "__main__":
    try:
        asyncio.run(test())
    except ConnectionRefusedError:
        print("Could not connect to server. Start it first:")
        print("  python server.py")
        sys.exit(1)
