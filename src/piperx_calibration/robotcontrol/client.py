from __future__ import annotations

import threading
import time
import io
import json
import zipfile
from collections.abc import Callable
from typing import Any

import httpx

from .camera import CalibrationFrame, decode_calibration_frame


class HeartbeatError(RuntimeError):
    """The control lease can no longer be kept alive."""


class RobotControlClient:
    """Small HTTP client used by VLA inference and calibration, never by the driver."""

    def __init__(self, base_url: str, *, http_client: httpx.Client | None = None):
        # robotcontrol is a loopback/LAN service. Download proxies from the
        # host environment must never intercept motion or camera traffic.
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(
            base_url=self._base_url, timeout=10.0, trust_env=False,
        )
        self._owns_client = http_client is None
        self.token: str | None = None
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: HeartbeatError | None = None
        self._heartbeat_lock = threading.Lock()

    def close(self) -> None:
        self.stop_heartbeat()
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            raise httpx.HTTPStatusError(
                f"{exc}; robotcontrol response: {detail or '<empty>'}",
                request=response.request,
                response=response,
            ) from exc
        return response.json() if response.content else None

    def acquire(self, owner_id: str) -> dict[str, Any]:
        payload = self._request("POST", "/v1/control/acquire", json={"owner_id": owner_id})
        self.token = payload["token"]
        with self._heartbeat_lock:
            self._heartbeat_error = None
        return payload

    def alive(self) -> dict[str, Any]:
        return self._request("POST", "/v1/control/alive", json={"token": self._require_token()})

    def release(self) -> None:
        if self.token is not None:
            self._request("POST", "/v1/control/release", json={"token": self.token})
            self.token = None

    def execute_joint_waypoints(
        self, waypoints: list[list[float]], profile: str = "vla_conservative", *, speed_scale: float | None = None,
    ) -> dict[str, Any]:
        self.raise_if_heartbeat_failed()
        payload = {
            "token": self._require_token(), "waypoints": waypoints, "profile": profile,
        }
        if speed_scale is not None:
            if speed_scale <= 0 or speed_scale > 2:
                raise ValueError("speed_scale must be in (0, 2]")
            payload["speed_scale"] = float(speed_scale)
        return self._request("POST", "/v1/motions/execute_joint_waypoints", json=payload)

    def arm_motion(self) -> None:
        self.raise_if_heartbeat_failed()
        self._request("POST", "/v1/motions/arm", json={"token": self._require_token()})

    def go_home(self) -> dict[str, Any]:
        self.raise_if_heartbeat_failed()
        return self._request("POST", "/v1/motions/go_home", json={"token": self._require_token()})

    def hold(self) -> None:
        self._request("POST", "/v1/motions/hold")

    def health(self) -> dict[str, Any]:
        """Read service health without acquiring a lease or touching hardware."""
        return self._request("GET", "/v1/health")

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/v1/state")

    def execution(self, execution_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/executions/{execution_id}")

    def wait_execution(self, execution_id: str, *, timeout_s: float = 30.0, poll_s: float = 0.05) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.execution(execution_id)
            if result["state"] != "running":
                return result
            time.sleep(poll_s)
        raise TimeoutError(f"execution {execution_id} did not finish within {timeout_s}s")

    def frame_jpeg(self, camera_id: str) -> tuple[bytes, float | None]:
        response = self._client.get(f"/v1/cameras/{camera_id}/frame")
        response.raise_for_status()
        timestamp = response.headers.get("x-frame-timestamp-monotonic-s")
        return response.content, float(timestamp) if timestamp is not None else None

    def frame_jpegs(self) -> dict[str, tuple[bytes, float | None]]:
        response = self._client.get("/v1/cameras/frame-batch")
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            return {
                camera_id: (
                    archive.read(entry["file"]),
                    float(entry["timestamp_monotonic_s"]),
                )
                for camera_id, entry in metadata["cameras"].items()
            }

    def calibration_frame(self, camera_id: str) -> tuple[CalibrationFrame, bytes]:
        response = self._client.get(f"/v1/cameras/{camera_id}/calibration-frame")
        response.raise_for_status()
        return decode_calibration_frame(response.content), response.content

    def set_camera_controls(self, camera_id: str, *, exposure: float, gain: float) -> dict[str, Any]:
        return self._request("POST", f"/v1/cameras/{camera_id}/controls", json={
            "exposure": exposure, "gain": gain,
        })

    def _heartbeat_once(self, client: httpx.Client) -> None:
        response = client.post("/v1/control/alive", json={"token": self._require_token()})
        response.raise_for_status()

    def raise_if_heartbeat_failed(self) -> None:
        with self._heartbeat_lock:
            error = self._heartbeat_error
        if error is not None:
            raise error

    def start_heartbeat(
        self,
        interval_s: float = 3.0,
        *,
        failure_timeout_s: float = 12.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        if failure_timeout_s <= interval_s:
            raise ValueError("heartbeat failure timeout must exceed the interval")
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        stop = threading.Event()
        self._heartbeat_stop = stop
        with self._heartbeat_lock:
            self._heartbeat_error = None

        def loop() -> None:
            last_success = time.monotonic()
            delay = interval_s
            with httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(2.0),
                trust_env=False,
            ) as heartbeat_client:
                while not stop.wait(delay):
                    try:
                        self._heartbeat_once(heartbeat_client)
                        last_success = time.monotonic()
                        delay = interval_s
                    except Exception as exc:
                        forbidden = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403
                        timed_out = time.monotonic() - last_success >= failure_timeout_s
                        if not forbidden and not timed_out:
                            delay = min(0.5, interval_s)
                            continue
                        error = HeartbeatError(
                            f"robotcontrol lease heartbeat failed: {type(exc).__name__}: {exc}"
                        )
                        with self._heartbeat_lock:
                            self._heartbeat_error = error
                        if on_error is not None:
                            on_error(error)
                        return

        self._heartbeat_thread = threading.Thread(target=loop, name="robotcontrol-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_stop = self._heartbeat_thread = None

    def _require_token(self) -> str:
        if self.token is None:
            raise RuntimeError("acquire control before sending motion commands")
        return self.token
