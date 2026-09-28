"""Render timings for work sent to a remote GPU worker
(docs/adr/render-time-estimate.md, "Remote workers").

When the GPU picker sends a render to an enrolled worker, what the user waits
for is the worker's GPU plus everything between the two machines: the task and
its reference audio going out, queueing on the worker, a model load when the
engine is not resident there, and the audio coming back. None of that is the
local machine's speed, so the control plane (this machine) times each remote
synthesis task end to end, from dispatch to the result read back here, and
files the row under the worker's own bucket::

    device = "remote:<worker id>:<gpu>"      e.g. remote:cbd2b78f7ddc:rtx-5080

A remote task is a whole unit of work, not one engine call: ``/generate``
sends the whole take (the worker splits it into chunks itself) and Audiobook
and Stories send one chapter per task. So a remote row is one task, and the
estimate plans a remote render as one call per task. Local and remote rows
never mix: every read is keyed by one device, and the estimate reads the
bucket of the target the render will actually go to
(:func:`worker.routing.status`). A chosen worker that cannot take work gets
no number at all, only a note that it is unavailable.

Same rule as every other row: numbers and identifiers only (the worker id is
the opaque 12-hex id the registry minted, the GPU its model name). Recording
never breaks a render: every failure here is logged at debug level and dropped.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from services import render_timing, render_warmth

logger = logging.getLogger("omnivoice.render_remote")

LOCAL = "local"
REMOTE = "remote"
UNAVAILABLE = "unavailable"

#: Vendor words that name no model: "NVIDIA GeForce RTX 5080" -> "rtx-5080".
_GPU_NOISE = re.compile(r"\b(nvidia|geforce|amd|corporation|graphics|gpu)\b")
_GPU_LIMIT = 24
_LABEL_LIMIT = 64


@dataclass(frozen=True)
class Route:
    """Where a render will run, as the estimate needs to know it."""

    kind: str = LOCAL
    worker_id: str = ""
    label: str = ""
    device: str = ""
    #: True when the chosen worker is not connected (vs. connected but paused,
    #: disabled or shutting down).
    offline: bool = False

    @property
    def remote(self) -> bool:
        return self.kind == REMOTE

    def to_dict(self) -> dict:
        """The response's ``target``: never the worker id, only its name."""
        if self.kind == LOCAL:
            return {"kind": LOCAL}
        out = {"kind": self.kind, "label": self.label[:_LABEL_LIMIT]}
        if self.kind == UNAVAILABLE:
            out["offline"] = self.offline
        return out


LOCAL_ROUTE = Route()


def gpu_slug(name: Any) -> str:
    """A short, stable id for a GPU model name ("" when unknown)."""
    text = _GPU_NOISE.sub(" ", str(name or "").lower())
    return render_timing.clean_id(" ".join(text.split()), limit=_GPU_LIMIT)


def remote_device(worker_id: Any, gpu_name: Any = "") -> str:
    """The render_timings device for work run on this worker."""
    worker = render_timing.clean_id(worker_id, limit=32)
    if not worker:
        return ""
    gpu = gpu_slug(gpu_name)
    return f"remote:{worker}:{gpu}" if gpu else f"remote:{worker}"


def engine_for(engine_id: Any) -> str:
    """The engine a remote row is filed under: the engine id the task names.
    (The local bucket may add a local adapter's model; a worker's configuration
    is not known here, so remote buckets never do.)"""
    return render_timing.clean_id(engine_id)


def _plane(control_plane=None):
    if control_plane is not None:
        return control_plane
    from worker.service import control_plane as default_plane

    return default_plane


def route_from_status(state: dict) -> Route:
    """Read :func:`worker.routing.status` for one operation."""
    from worker import routing

    active = state.get("active") or {}
    targets = {t.get("id"): t for t in state.get("targets") or [] if isinstance(t, dict)}
    if active.get("remote"):
        worker_id = str(active.get("worker_id") or "")
        entry = targets.get(worker_id) or {}
        device = remote_device(worker_id, entry.get("gpu_name"))
        if not device:
            return LOCAL_ROUTE
        return Route(REMOTE, worker_id=worker_id,
                     label=str(active.get("label") or entry.get("label") or ""), device=device)
    chosen = str(state.get("target") or routing.LOCAL)
    op = state.get("op") or ""
    if chosen == routing.LOCAL or (op and op not in (state.get("remote_operations") or ())):
        return LOCAL_ROUTE
    entry = targets.get(chosen)
    if entry is None:
        # The chosen worker no longer exists: routing runs the work here.
        return LOCAL_ROUTE
    return Route(UNAVAILABLE, worker_id=chosen, label=str(entry.get("label") or ""),
                 offline=entry.get("status") == "offline" or not entry.get("connected"))


def route_for(op: str, *, control_plane=None) -> Route:
    """The route a render of ``op`` would take right now. Never raises: when
    routing cannot be read, the render would run here too."""
    try:
        from worker import routing

        return route_from_status(routing.status(_plane(control_plane), op=op))
    except Exception:  # noqa: BLE001 - routing is advisory; local always works
        logger.debug("render route unavailable", exc_info=True)
        return LOCAL_ROUTE


def route_from_decision(decision: Any, *, control_plane=None) -> Route:
    """The route of a render whose routing was already decided (a running
    render never re-asks: the user may switch targets mid-render)."""
    if not getattr(decision, "remote", False):
        return LOCAL_ROUTE
    worker_id = str(getattr(decision, "worker_id", "") or "")
    device = device_for(worker_id, control_plane=control_plane)
    if not device:
        return LOCAL_ROUTE
    return Route(REMOTE, worker_id=worker_id, label=str(getattr(decision, "label", "") or ""),
                 device=device)


def device_for(worker_id: str, *, control_plane=None) -> str:
    """The bucket device of an enrolled worker (its GPU from the registry)."""
    gpu = ""
    try:
        from worker import routing

        for target in routing.list_targets(_plane(control_plane)):
            if target.id == worker_id:
                gpu = target.gpu_name
                break
    except Exception:  # noqa: BLE001
        logger.debug("worker GPU unavailable", exc_info=True)
    return remote_device(worker_id, gpu)


def worker_resident(worker_id: str, engine: str, *, control_plane=None) -> Optional[bool]:
    """Whether the worker has ``engine`` loaded in VRAM right now, from its
    heartbeats; ``None`` when that is not known."""
    try:
        plane = _plane(control_plane)
        pool = getattr(plane, "pool", None) if getattr(plane, "running", False) else None
        live = pool.get(worker_id) if pool is not None else None
        capacity = getattr(live, "capacity", None)
        if capacity is None:
            return None
        for cap in getattr(live.record, "capabilities", None) or []:
            model_id = str(cap.get("model_id") or "")
            if cap.get("engine") == engine and model_id:
                return bool(capacity.is_resident(engine, model_id))
    except Exception:  # noqa: BLE001
        logger.debug("worker residency unavailable", exc_info=True)
    return None


def audio_seconds_of(path: Optional[str]) -> Optional[float]:
    """Length of a result artifact, from its header only."""
    if not path:
        return None
    try:
        import soundfile as sf

        seconds = float(sf.info(path).duration)
    except Exception:  # noqa: BLE001
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _steps(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


class RemoteTimer:
    """Times one remote task from dispatch to the result read back here."""

    def __init__(self, *, engine: str, device_of: Callable[[], str], num_step: Optional[int],
                 speed: Any, text_chars: int, cold: Optional[str]):
        self.engine = engine
        self._device_of = device_of
        self.num_step = num_step
        self.speed = speed
        self.text_chars = text_chars
        self.cold = cold
        self._started: Optional[float] = None

    def start(self) -> None:
        self._started = time.perf_counter()

    def watch(self, on_state: Optional[Callable[[dict], None]]) -> Callable[[dict], None]:
        """``on_state`` for the gateway: notes a model load on the worker
        (the task passed through ``model_loading``), then forwards."""
        from services.gpu_gateway import PHASE_LOADING

        def _listen(payload: dict) -> None:
            if isinstance(payload, dict) and payload.get("phase") == PHASE_LOADING:
                self.cold = render_warmth.LOAD
            if on_state is not None:
                on_state(payload)

        return _listen

    def finish(self, task: Any) -> bool:
        """The task's result was read back: record the row. Never raises."""
        try:
            if self._started is None:
                return False
            wall = time.perf_counter() - self._started
            audio = audio_seconds_of(getattr(task, "result_ref", None))
            device = self._device_of()
            if audio is None or not device:
                return False
            return render_timing.record(self.engine, device, self.num_step, self.text_chars,
                                        audio, wall, self.speed, None, self.cold)
        except Exception:  # noqa: BLE001 - a timing row is never worth a failed render
            logger.debug("remote render timing not recorded", exc_info=True)
            return False


def begin(call: Any, decision: Any, *, control_plane=None) -> Optional[RemoteTimer]:
    """A timer for ``call`` when it carries ``timed`` (the render surfaces
    set it), else None. Reads the worker's residency BEFORE dispatch: the
    task itself loads the model. Never raises."""
    try:
        timed = getattr(call, "timed", None)
        worker_id = str(getattr(decision, "worker_id", "") or "")
        if not isinstance(timed, dict) or not worker_id:
            return None
        engine = engine_for(getattr(call, "engine", ""))
        if not engine:
            return None
        text = (getattr(call, "params", None) or {}).get("text")
        resident = worker_resident(worker_id, str(getattr(call, "engine", "")),
                                   control_plane=control_plane)
        return RemoteTimer(
            engine=engine,
            device_of=lambda: device_for(worker_id, control_plane=control_plane),
            num_step=_steps(timed.get("num_step")),
            speed=timed.get("speed"),
            text_chars=len(text) if isinstance(text, str) else 0,
            cold=render_warmth.LOAD if resident is False else None,
        )
    except Exception:  # noqa: BLE001
        logger.debug("remote render timing unavailable", exc_info=True)
        return None


__all__ = [
    "LOCAL", "LOCAL_ROUTE", "REMOTE", "RemoteTimer", "Route", "UNAVAILABLE",
    "audio_seconds_of", "begin", "device_for", "engine_for", "gpu_slug", "remote_device",
    "route_for", "route_from_decision", "route_from_status", "worker_resident",
]
