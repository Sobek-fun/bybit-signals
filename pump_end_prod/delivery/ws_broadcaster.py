import asyncio
from threading import Thread
from typing import Dict, Set

import websockets


class WsBroadcaster:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients_by_path: Dict[str, Set] = {}
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
        path = websocket.path if hasattr(websocket, 'path') else '/ws/all'
        if path not in self.clients_by_path:
            self.clients_by_path[path] = set()
        self.clients_by_path[path].add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients_by_path[path].discard(websocket)

    def broadcast(self, payload: str, source: str = None):
        if not self.loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload, source), self.loop)

    async def _broadcast(self, payload: str, source: str = None):
        paths_to_send = ['/ws/all']
        if source:
            paths_to_send.append(f'/ws/{source}')

        dead_clients = set()
        for path in paths_to_send:
            clients = self.clients_by_path.get(path, set())
            for client in clients:
                try:
                    await client.send(payload)
                except:
                    dead_clients.add((path, client))

        for path, client in dead_clients:
            if path in self.clients_by_path:
                self.clients_by_path[path].discard(client)

    def stop(self):
        if self.server:
            self.server.close()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
