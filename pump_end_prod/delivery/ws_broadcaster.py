import asyncio
from threading import Thread
from typing import Set

import websockets


class WsBroadcaster:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.loop = None
        self.server = None
        self.thread = Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())
        self.loop.run_forever()

    async def _start(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    def broadcast(self, payload: str):
        if not self.loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, payload: str):
        if not self.clients:
            return
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send(payload)
            except:
                dead_clients.add(client)
        self.clients -= dead_clients

    def stop(self):
        if self.server:
            self.server.close()
        if self.loop:
            self.loop.stop()
