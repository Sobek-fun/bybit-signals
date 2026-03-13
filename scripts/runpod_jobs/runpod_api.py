from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class RunpodError(RuntimeError):
    pass


@dataclass(slots=True)
class PodSshEndpoint:
    pod_id: str
    host: str
    port: int
    desired_status: str


class RunpodClient:
    def __init__(self, api_key: str, base_url: str = "https://rest.runpod.io/v1"):
        self.api_key = api_key.strip()
        if not self.api_key:
            raise RunpodError("RunPod API key is required")
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            response = requests.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=45,
            )
        except requests.RequestException as exc:
            raise RunpodError(f"RunPod network error: {exc}") from exc
        if response.status_code >= 400:
            raise RunpodError(f"RunPod {method} {path} failed: {response.status_code} {response.text[:500]}")
        if response.status_code == 204:
            return None
        return response.json()

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/pods/{pod_id}")
        if not isinstance(payload, dict):
            raise RunpodError(f"Unexpected pod payload: {payload}")
        return payload

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        created = self._request("POST", "/pods", payload)
        if not isinstance(created, dict) or "id" not in created:
            raise RunpodError(f"Unexpected create_pod payload: {created}")
        return created

    def wait_for_ssh_endpoint(self, pod_id: str, timeout_seconds: int = 1200, poll_seconds: int = 8) -> PodSshEndpoint:
        deadline = time.time() + timeout_seconds
        last: dict[str, Any] = {}
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod
            desired = str(pod.get("desiredStatus") or "")
            ip = pod.get("publicIp")
            port_map = pod.get("portMappings") or {}
            ssh_port = None
            if isinstance(port_map, dict):
                ssh_port = port_map.get("22")
                if ssh_port is None:
                    ssh_port = port_map.get(22)
            if desired == "RUNNING" and ip and ssh_port:
                return PodSshEndpoint(
                    pod_id=pod_id,
                    host=str(ip),
                    port=int(ssh_port),
                    desired_status=desired,
                )
            time.sleep(poll_seconds)
        raise RunpodError(f"Timed out waiting pod SSH endpoint, last={last}")
