"""Access event log.

Events carry the decision and the evidence for it, never embeddings or imagery,
so the log can be shipped off the device without moving biometric data.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessEvent:
    decision: str
    reason: str
    identity: str = ""
    similarity: float = 0.0
    texture_score: float = 0.0
    blink_count: int = 0
    device_id: str = "flock-door"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class EventSink(Protocol):
    def emit(self, event: AccessEvent) -> None: ...


class JsonlFileSink:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: AccessEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.as_json() + "\n")


class CloudWatchLogsSink:
    """Ships events to CloudWatch Logs. Requires the ``aws`` extra."""

    def __init__(self, log_group: str, log_stream: str, client=None) -> None:
        if client is None:
            import boto3

            client = boto3.client("logs")
        self._client = client
        self._group = log_group
        self._stream = log_stream
        self._ensure_stream()

    def _ensure_stream(self) -> None:
        for create, kwargs in (
            (self._client.create_log_group, {"logGroupName": self._group}),
            (
                self._client.create_log_stream,
                {"logGroupName": self._group, "logStreamName": self._stream},
            ),
        ):
            try:
                create(**kwargs)
            except Exception as exc:
                if "ResourceAlreadyExists" not in type(exc).__name__:
                    logger.debug("log setup: %s", exc)

    def emit(self, event: AccessEvent) -> None:
        self._client.put_log_events(
            logGroupName=self._group,
            logStreamName=self._stream,
            logEvents=[
                {
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "message": event.as_json(),
                },
            ],
        )


class MultiSink:
    """Fans out to several sinks. A failing remote sink must not block the door."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = list(sinks)

    def emit(self, event: AccessEvent) -> None:
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                logger.exception("event sink failed: %s", type(sink).__name__)
