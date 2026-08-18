import json

from flock.events import AccessEvent, JsonlFileSink, MultiSink


class BrokenSink:
    def emit(self, event):
        msg = "remote unavailable"
        raise RuntimeError(msg)


def test_event_serialises_to_one_line(tmp_path):
    sink = JsonlFileSink(tmp_path / "events.jsonl")
    sink.emit(AccessEvent(decision="unlock", reason="match", identity="alice"))
    sink.emit(AccessEvent(decision="deny", reason="unknown_identity"))
    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["identity"] == "alice"


def test_failing_sink_does_not_block_the_door(tmp_path):
    local = JsonlFileSink(tmp_path / "events.jsonl")
    MultiSink(BrokenSink(), local).emit(AccessEvent(decision="unlock", reason="match"))
    assert (tmp_path / "events.jsonl").exists()


def test_event_has_timestamp_and_no_biometrics():
    payload = json.loads(AccessEvent(decision="deny", reason="no_face").as_json())
    assert payload["timestamp"].endswith("+00:00")
    assert set(payload) == {
        "decision", "reason", "identity", "similarity",
        "texture_score", "blink_count", "device_id", "timestamp",
    }
