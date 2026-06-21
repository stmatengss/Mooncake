#!/usr/bin/env python3
"""Minimal HTTP metadata server for transfer-engine integration tests."""

import asyncio
import os
import sys
from aiohttp import web


class MetadataServer:
    def __init__(self, port: int):
        self.port = port
        self.store = {}

    async def handle_metadata(self, request: web.Request) -> web.Response:
        key = request.query.get("key", "")
        if request.method == "GET":
            value = self.store.get(key)
            if value is None:
                return web.Response(text="metadata not found", status=404)
            return web.Response(body=value, status=200)
        if request.method == "PUT":
            body = await request.read()
            if "rpc_meta" in key and key in self.store:
                if self.store[key] == body:
                    return web.Response(text="metadata unchanged", status=200)
                return web.Response(
                    text="Duplicate rpc_meta key not allowed", status=400
                )
            self.store[key] = body
            return web.Response(text="metadata updated", status=200)
        if request.method == "DELETE":
            if key not in self.store:
                return web.Response(text="metadata not found", status=404)
            del self.store[key]
            return web.Response(text="metadata deleted", status=200)
        return web.Response(text="Method not allowed", status=405)


async def main() -> None:
    port = int(sys.argv[1])
    server = MetadataServer(port)
    app = web.Application()
    app.router.add_route("*", "/metadata", server.handle_metadata)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"READY {port}", flush=True)
    # Detach stdout so popen-based test fixtures cannot deadlock if aiohttp
    # or asyncio writes logs after startup.
    sys.stdout = open(os.devnull, "w")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
