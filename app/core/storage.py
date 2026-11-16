"""Small SQLite event store with explicit JSON serialization."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import EventListResponse, EventSummary, PredictionEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_events (
    event_id TEXT PRIMARY KEY,
    point_id TEXT,
    timestamp TEXT NOT NULL,
    input_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    detections_json TEXT NOT NULL,
    inspection_required INTEGER NOT NULL CHECK (inspection_required IN (0, 1)),
    status TEXT NOT NULL,
    evidence_uri TEXT,
    inference_ms REAL NOT NULL CHECK (inference_ms >= 0),
    disclaimer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_prediction_events_timestamp
ON prediction_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_prediction_events_point
ON prediction_events(point_id);
"""


class EventRepository:
    """Persist and query prediction events without storing uploaded originals."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def save(self, event: PredictionEvent) -> None:
        payload = event.model_dump(mode="json")
        detections_json = json.dumps(
            payload["detections"], ensure_ascii=False, separators=(",", ":")
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    event_id, point_id, timestamp, input_type, model_version,
                    detections_json, inspection_required, status, evidence_uri,
                    inference_ms, disclaimer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["event_id"],
                    payload["point_id"],
                    payload["timestamp"],
                    payload["input_type"],
                    payload["model_version"],
                    detections_json,
                    int(payload["inspection_recommended"]),
                    payload["status"],
                    payload["evidence_uri"],
                    payload["inference_ms"],
                    payload["disclaimer"],
                ),
            )

    def get(self, event_id: str) -> PredictionEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM prediction_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._event_from_row(row) if row is not None else None

    def list(
        self, *, limit: int = 50, offset: int = 0, point_id: str | None = None
    ) -> EventListResponse:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        where = ""
        parameters: list[object] = []
        if point_id is not None:
            where = " WHERE point_id = ?"
            parameters.append(point_id)

        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM prediction_events{where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT event_id, point_id, timestamp, input_type, model_version, "
                "detections_json, inspection_required, status, evidence_uri "
                f"FROM prediction_events{where} "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()

        items: list[EventSummary] = []
        for row in rows:
            detections = json.loads(row["detections_json"])
            confidences = [float(item["confidence"]) for item in detections]
            items.append(
                EventSummary(
                    event_id=row["event_id"],
                    point_id=row["point_id"],
                    timestamp=row["timestamp"],
                    input_type=row["input_type"],
                    model_version=row["model_version"],
                    detection_count=len(detections),
                    detected_classes=sorted(
                        {str(item["class_name"]) for item in detections}
                    ),
                    max_confidence=max(confidences, default=None),
                    inspection_recommended=bool(row["inspection_required"]),
                    status=row["status"],
                    evidence_uri=row["evidence_uri"],
                )
            )
        return EventListResponse(items=items, total=total)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> PredictionEvent:
        return PredictionEvent.model_validate(
            {
                "event_id": row["event_id"],
                "point_id": row["point_id"],
                "timestamp": row["timestamp"],
                "input_type": row["input_type"],
                "model_version": row["model_version"],
                "detections": json.loads(row["detections_json"]),
                "inspection_recommended": bool(row["inspection_required"]),
                "status": row["status"],
                "evidence_uri": row["evidence_uri"],
                "inference_ms": row["inference_ms"],
                "disclaimer": row["disclaimer"],
            }
        )


class EvidenceStore:
    """Store derived annotated evidence inside one configured directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def write(self, event_id: str, data: bytes, *, suffix: str = ".jpg") -> str:
        if not event_id or any(char not in "0123456789abcdef-" for char in event_id):
            raise ValueError("invalid event id")
        if suffix not in {".jpg", ".json"}:
            raise ValueError("unsupported evidence suffix")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = (self.root / f"{event_id}{suffix}").resolve()
        destination.relative_to(self.root)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        return destination.relative_to(self.root).as_posix()

    def resolve(self, relative_uri: str) -> Path:
        candidate = (self.root / relative_uri).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("evidence path leaves configured directory") from exc
        return candidate

    def delete(self, relative_uri: str) -> None:
        """Remove one derived evidence file without allowing path traversal."""

        self.resolve(relative_uri).unlink(missing_ok=True)
