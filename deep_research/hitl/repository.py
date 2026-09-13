"""SQLite persistence for durable HITL interactions."""

from __future__ import annotations

import sqlite3
import threading
import json
from uuid import uuid4
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .models import HITLAction, HITLInteraction, HITLStatus


class HITLRepositoryError(Exception):
    """Base exception for HITL repository failures."""


class RepositoryClosedError(HITLRepositoryError):
    """Raised when an operation is attempted after repository close."""


class DuplicateInteractionError(HITLRepositoryError):
    """Raised when an interaction violates a uniqueness rule."""


class ActiveThreadInteractionError(HITLRepositoryError):
    """Raised when a thread already has an unresolved HITL interaction."""

    def __init__(self, interaction: HITLInteraction) -> None:
        self.interaction = interaction
        super().__init__(
            "thread already has an unresolved HITL interaction: "
            f"{interaction.interaction_id}"
        )


class InteractionNotFoundError(HITLRepositoryError):
    """Raised when a requested interaction does not exist."""


_ACTION_VALUES = tuple(action.value for action in HITLAction)
_STATUS_VALUES = tuple(status.value for status in HITLStatus)
_BLOCKING_STATUS_VALUES = tuple(
    status.value
    for status in HITLStatus
    if status
    not in {
        HITLStatus.RESUMED,
        HITLStatus.STALE,
        HITLStatus.EXPIRED,
    }
)


def _timestamp_to_storage(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _timestamp_from_storage(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _interaction_to_parameters(
    interaction: HITLInteraction,
) -> tuple[object, ...]:
    return (
        interaction.interaction_id,
        interaction.thread_id,
        interaction.run_id,
        interaction.interrupt_id,
        interaction.checkpoint_id,
        interaction.action.value,
        interaction.target_type,
        interaction.target_id,
        interaction.target_version,
        interaction.status.value,
        _timestamp_to_storage(interaction.expires_at),
        _timestamp_to_storage(interaction.created_at),
        _timestamp_to_storage(interaction.updated_at),
        _timestamp_to_storage(interaction.resolved_at),
        interaction.decision_actor,
        interaction.decision_reason,
        interaction.error_code,
    )


def _row_to_interaction(row: sqlite3.Row) -> HITLInteraction:
    return HITLInteraction(
        interaction_id=row["interaction_id"],
        thread_id=row["thread_id"],
        run_id=row["run_id"],
        interrupt_id=row["interrupt_id"],
        checkpoint_id=row["checkpoint_id"],
        action=HITLAction(row["action"]),
        target_type=row["target_type"],
        target_id=row["target_id"],
        target_version=row["target_version"],
        status=HITLStatus(row["status"]),
        expires_at=_timestamp_from_storage(row["expires_at"]),
        created_at=_timestamp_from_storage(row["created_at"]),
        updated_at=_timestamp_from_storage(row["updated_at"]),
        resolved_at=_timestamp_from_storage(row["resolved_at"]),
        decision_actor=row["decision_actor"],
        decision_reason=row["decision_reason"],
        error_code=row["error_code"],
    )


class HITLRepository:
    """Thread-safe, explicitly opened SQLite repository."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def initialize(self) -> None:
        """Open the database and create the schema if it is absent."""

        with self._lock:
            if self._connection is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                connection.executescript(
                    f"""
                    CREATE TABLE IF NOT EXISTS hitl_interactions (
                        interaction_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        interrupt_id TEXT,
                        checkpoint_id TEXT,
                        action TEXT NOT NULL CHECK (
                            action IN ({','.join(repr(value) for value in _ACTION_VALUES)})
                        ),
                        target_type TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        target_version INTEGER NOT NULL CHECK (
                            target_version >= 1
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN ({','.join(repr(value) for value in _STATUS_VALUES)})
                        ),
                        expires_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        resolved_at TEXT,
                        decision_actor TEXT,
                        decision_reason TEXT,
                        error_code TEXT,
                        UNIQUE (
                            run_id,
                            action,
                            target_type,
                            target_id,
                            target_version
                        )
                    );

                    CREATE INDEX IF NOT EXISTS
                        idx_hitl_interactions_thread_status
                    ON hitl_interactions (thread_id, status, updated_at DESC);

                    CREATE INDEX IF NOT EXISTS
                        idx_hitl_interactions_run
                    ON hitl_interactions (run_id, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS hitl_idempotency_keys (
                        interaction_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        status TEXT NOT NULL,
                        response_json TEXT,
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (interaction_id, idempotency_key)
                    );

                    CREATE TABLE IF NOT EXISTS hitl_audit_events (
                        audit_id TEXT PRIMARY KEY,
                        interaction_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        actor TEXT,
                        decision TEXT,
                        idempotency_key TEXT,
                        status TEXT,
                        error_code TEXT,
                        interrupt_id TEXT,
                        checkpoint_id TEXT,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_hitl_audit_interaction
                    ON hitl_audit_events (interaction_id, created_at DESC);
                    """
                )
                self._migrate_schema(connection)
                connection.commit()
            except Exception:
                connection.close()
                raise

            self._connection = connection

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Upgrade databases created before runtime binding was persisted."""

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(hitl_interactions)"
            ).fetchall()
        }
        if "interrupt_id" not in columns:
            connection.execute(
                "ALTER TABLE hitl_interactions ADD COLUMN interrupt_id TEXT"
            )
        if "checkpoint_id" not in columns:
            connection.execute(
                "ALTER TABLE hitl_interactions ADD COLUMN checkpoint_id TEXT"
            )
        if "expires_at" not in columns:
            connection.execute(
                "ALTER TABLE hitl_interactions ADD COLUMN expires_at TEXT"
            )

        table_row = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'hitl_interactions'
            """
        ).fetchone()
        table_sql = str(table_row[0] or "").lower() if table_row else ""
        if "'stale'" not in table_sql:
            connection.execute(
                "DROP INDEX IF EXISTS idx_hitl_interactions_thread_status"
            )
            connection.execute("DROP INDEX IF EXISTS idx_hitl_interactions_run")
            connection.execute(
                "ALTER TABLE hitl_interactions RENAME TO hitl_interactions_legacy"
            )
            connection.executescript(
                f"""
                CREATE TABLE hitl_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    interrupt_id TEXT,
                    checkpoint_id TEXT,
                    action TEXT NOT NULL CHECK (
                        action IN ({','.join(repr(value) for value in _ACTION_VALUES)})
                    ),
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_version INTEGER NOT NULL CHECK (target_version >= 1),
                    status TEXT NOT NULL CHECK (
                        status IN ({','.join(repr(value) for value in _STATUS_VALUES)})
                    ),
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT,
                    decision_actor TEXT,
                    decision_reason TEXT,
                    error_code TEXT,
                    UNIQUE (
                        run_id, action, target_type, target_id, target_version
                    )
                );

                INSERT INTO hitl_interactions (
                    interaction_id, thread_id, run_id, action, target_type,
                    target_id, target_version, status, created_at, updated_at,
                    expires_at, resolved_at, decision_actor, decision_reason, error_code,
                    interrupt_id, checkpoint_id
                )
                SELECT interaction_id, thread_id, run_id, action, target_type,
                    target_id, target_version, status, created_at, updated_at, expires_at,
                    resolved_at, decision_actor, decision_reason, error_code,
                    interrupt_id, checkpoint_id
                FROM hitl_interactions_legacy;

                DROP TABLE hitl_interactions_legacy;
                """
            )

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_hitl_interactions_thread_status
            ON hitl_interactions (thread_id, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_hitl_interactions_run
            ON hitl_interactions (run_id, updated_at DESC);
            """
        )

    def close(self) -> None:
        """Close the SQLite connection; repeated close calls are safe."""

        with self._lock:
            if self._connection is None:
                return

            self._connection.close()
            self._connection = None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_connection()
        with self._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RepositoryClosedError("HITL repository is not open")
        return self._connection

    def create(self, interaction: HITLInteraction) -> HITLInteraction:
        """Persist one interaction in a transaction."""

        with self._transaction() as connection:
            try:
                # Preserve the repository's request-key contract.  A retry
                # for the exact same request must surface as a duplicate so
                # the service layer can resolve it idempotently, even when a
                # previous interaction is still blocking the thread.
                duplicate = connection.execute(
                    """
                    SELECT 1
                    FROM hitl_interactions
                    WHERE run_id = ?
                      AND action = ?
                      AND target_type = ?
                      AND target_id = ?
                      AND target_version IS ?
                    LIMIT 1
                    """,
                    (
                        interaction.run_id,
                        interaction.action.value,
                        interaction.target_type,
                        interaction.target_id,
                        interaction.target_version,
                    ),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateInteractionError(
                        "interaction request key already exists"
                    )

                placeholders = ", ".join("?" for _ in _BLOCKING_STATUS_VALUES)
                blocking = connection.execute(
                    f"""
                    SELECT *
                    FROM hitl_interactions
                    WHERE thread_id = ?
                      AND status IN ({placeholders})
                    ORDER BY updated_at DESC, interaction_id ASC
                    LIMIT 1
                    """,
                    (interaction.thread_id, *_BLOCKING_STATUS_VALUES),
                ).fetchone()
                if blocking is not None:
                    raise ActiveThreadInteractionError(
                        _row_to_interaction(blocking)
                    )

                connection.execute(
                    """
                    INSERT INTO hitl_interactions (
                        interaction_id,
                        thread_id,
                        run_id,
                        interrupt_id,
                        checkpoint_id,
                        action,
                        target_type,
                        target_id,
                        target_version,
                        status,
                        expires_at,
                        created_at,
                        updated_at,
                        resolved_at,
                        decision_actor,
                        decision_reason,
                        error_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _interaction_to_parameters(interaction),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateInteractionError(
                    "interaction id or request key already exists"
                ) from error

        return interaction

    def update(self, interaction: HITLInteraction) -> HITLInteraction:
        """Persist the current state of an existing interaction."""

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE hitl_interactions
                SET thread_id = ?,
                    run_id = ?,
                    interrupt_id = ?,
                    checkpoint_id = ?,
                    action = ?,
                    target_type = ?,
                    target_id = ?,
                    target_version = ?,
                    status = ?,
                    expires_at = ?,
                    created_at = ?,
                    updated_at = ?,
                    resolved_at = ?,
                    decision_actor = ?,
                    decision_reason = ?,
                    error_code = ?
                WHERE interaction_id = ?
                """,
                (
                    interaction.thread_id,
                    interaction.run_id,
                    interaction.interrupt_id,
                    interaction.checkpoint_id,
                    interaction.action.value,
                    interaction.target_type,
                    interaction.target_id,
                    interaction.target_version,
                    interaction.status.value,
                    _timestamp_to_storage(interaction.expires_at),
                    _timestamp_to_storage(interaction.created_at),
                    _timestamp_to_storage(interaction.updated_at),
                    _timestamp_to_storage(interaction.resolved_at),
                    interaction.decision_actor,
                    interaction.decision_reason,
                    interaction.error_code,
                    interaction.interaction_id,
                ),
            )
            if cursor.rowcount != 1:
                raise InteractionNotFoundError("HITL interaction not found")

        return interaction

    def claim_resume(
        self,
        interaction_id: str,
    ) -> tuple[HITLInteraction, bool]:
        """Atomically claim a resolved interaction for graph resumption.

        Two browser clicks must not both pass a read-then-write check and
        issue two ``Command(resume=...)`` calls for the same interrupt.
        SQLite's transaction lock plus the conditional UPDATE make the claim
        single-winner even when requests arrive almost simultaneously.
        """

        timestamp = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM hitl_interactions
                WHERE interaction_id = ?
                """,
                (interaction_id,),
            ).fetchone()
            if row is None:
                raise InteractionNotFoundError("HITL interaction not found")

            cursor = connection.execute(
                """
                UPDATE hitl_interactions
                SET status = ?, updated_at = ?, error_code = NULL
                WHERE interaction_id = ?
                  AND status IN (?, ?, ?)
                """,
                (
                    HITLStatus.RESUMING.value,
                    timestamp,
                    interaction_id,
                    HITLStatus.APPROVED.value,
                    HITLStatus.REJECTED.value,
                    HITLStatus.FAILED.value,
                ),
            )
            latest = connection.execute(
                """
                SELECT *
                FROM hitl_interactions
                WHERE interaction_id = ?
                """,
                (interaction_id,),
            ).fetchone()
            if latest is None:
                raise InteractionNotFoundError("HITL interaction not found")

            # A zero-row update is intentional: return the current state so
            # the service can make an idempotent/in-flight decision.
            return _row_to_interaction(latest), cursor.rowcount == 1

    def get(self, interaction_id: str) -> HITLInteraction:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM hitl_interactions
                WHERE interaction_id = ?
                """,
                (interaction_id,),
            ).fetchone()

        if row is None:
            raise InteractionNotFoundError("HITL interaction not found")
        return _row_to_interaction(row)

    def find_by_request_key(
        self,
        *,
        run_id: str,
        action: HITLAction,
        target_type: str,
        target_id: str,
        target_version: int,
    ) -> HITLInteraction | None:
        """Find an existing interaction for one deterministic request."""

        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT *
                FROM hitl_interactions
                WHERE run_id = ?
                  AND action = ?
                  AND target_type = ?
                  AND target_id = ?
                  AND target_version = ?
                """,
                (
                    run_id,
                    action.value,
                    target_type,
                    target_id,
                    target_version,
                ),
            ).fetchone()

        return _row_to_interaction(row) if row is not None else None

    def list_for_thread(
        self,
        thread_id: str,
        *,
        statuses: Iterable[HITLStatus] | None = None,
        limit: int = 100,
    ) -> list[HITLInteraction]:
        """List interactions for one thread without exposing other threads."""

        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        connection = self._require_connection()
        parameters: list[object] = [thread_id]
        status_clause = ""
        if statuses is not None:
            status_values = [
                status.value if isinstance(status, HITLStatus) else str(status)
                for status in statuses
            ]
            if not status_values:
                return []
            placeholders = ", ".join("?" for _ in status_values)
            status_clause = f"AND status IN ({placeholders})"
            parameters.extend(status_values)
        parameters.append(limit)

        with self._lock:
            rows = connection.execute(
                f"""
                SELECT *
                FROM hitl_interactions
                WHERE thread_id = ?
                  {status_clause}
                ORDER BY updated_at DESC, interaction_id ASC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()

        return [_row_to_interaction(row) for row in rows]

    def list_for_target(
        self,
        target_type: str,
        target_id: str,
        *,
        target_version: int | None = None,
        statuses: Iterable[HITLStatus] | None = None,
        limit: int = 100,
    ) -> list[HITLInteraction]:
        """List interactions attached to one durable target.

        Publishing-center decisions are addressed by approval id rather than
        thread id, so they need a target-scoped lookup to reconcile the
        conversation HITL record with the publishing database.
        """

        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        connection = self._require_connection()
        parameters: list[object] = [target_type, target_id]
        version_clause = ""
        if target_version is not None:
            version_clause = "AND target_version = ?"
            parameters.append(target_version)
        status_clause = ""
        if statuses is not None:
            status_values = [
                status.value if isinstance(status, HITLStatus) else str(status)
                for status in statuses
            ]
            if not status_values:
                return []
            placeholders = ", ".join("?" for _ in status_values)
            status_clause = f"AND status IN ({placeholders})"
            parameters.extend(status_values)
        parameters.append(limit)

        with self._lock:
            rows = connection.execute(
                f"""
                SELECT *
                FROM hitl_interactions
                WHERE target_type = ?
                  AND target_id = ?
                  {version_clause}
                  {status_clause}
                ORDER BY updated_at DESC, interaction_id ASC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()

        return [_row_to_interaction(row) for row in rows]

    def list_for_run(self, run_id: str) -> list[HITLInteraction]:
        """List all persisted interactions belonging to one Agent run."""

        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """
                SELECT *
                FROM hitl_interactions
                WHERE run_id = ?
                ORDER BY updated_at DESC, interaction_id ASC
                """,
                (run_id,),
            ).fetchall()

        return [_row_to_interaction(row) for row in rows]

    def get_idempotency(
        self,
        interaction_id: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """
                SELECT * FROM hitl_idempotency_keys
                WHERE interaction_id = ? AND idempotency_key = ?
                """,
                (interaction_id, idempotency_key),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_idempotency(
        self,
        *,
        interaction_id: str,
        idempotency_key: str,
        decision: str,
        now: datetime | None = None,
    ) -> dict[str, object]:
        timestamp = _timestamp_to_storage(now or datetime.now(timezone.utc))
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO hitl_idempotency_keys
                (interaction_id, idempotency_key, decision, status, created_at, updated_at)
                VALUES (?, ?, ?, 'started', ?, ?)
                """,
                (interaction_id, idempotency_key, decision, timestamp, timestamp),
            )
            row = connection.execute(
                """
                SELECT * FROM hitl_idempotency_keys
                WHERE interaction_id = ? AND idempotency_key = ?
                """,
                (interaction_id, idempotency_key),
            ).fetchone()
        if row is None:
            raise InteractionNotFoundError("HITL idempotency record not found")
        return dict(row)

    def update_idempotency(
        self,
        *,
        interaction_id: str,
        idempotency_key: str,
        status: str,
        response: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> None:
        timestamp = _timestamp_to_storage(datetime.now(timezone.utc))
        response_json = json.dumps(response, ensure_ascii=False) if response is not None else None
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE hitl_idempotency_keys
                SET status = ?, response_json = ?, error_code = ?, updated_at = ?
                WHERE interaction_id = ? AND idempotency_key = ?
                """,
                (status, response_json, error_code, timestamp, interaction_id, idempotency_key),
            )

    def append_audit(
        self,
        *,
        interaction_id: str,
        thread_id: str,
        event_type: str,
        actor: str | None = None,
        decision: str | None = None,
        idempotency_key: str | None = None,
        status: str | None = None,
        error_code: str | None = None,
        interrupt_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        audit_id = uuid4().hex
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO hitl_audit_events
                (audit_id, interaction_id, thread_id, event_type, actor, decision,
                 idempotency_key, status, error_code, interrupt_id, checkpoint_id,
                 metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    interaction_id,
                    thread_id,
                    event_type,
                    actor,
                    decision,
                    idempotency_key,
                    status,
                    error_code,
                    interrupt_id,
                    checkpoint_id,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    _timestamp_to_storage(datetime.now(timezone.utc)),
                ),
            )
        return audit_id

    def list_audit(self, interaction_id: str, *, limit: int = 100) -> list[dict[str, object]]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """
                SELECT * FROM hitl_audit_events
                WHERE interaction_id = ?
                ORDER BY created_at ASC, audit_id ASC
                LIMIT ?
                """,
                (interaction_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "ActiveThreadInteractionError",
    "DuplicateInteractionError",
    "HITLRepository",
    "HITLRepositoryError",
    "InteractionNotFoundError",
    "RepositoryClosedError",
]
