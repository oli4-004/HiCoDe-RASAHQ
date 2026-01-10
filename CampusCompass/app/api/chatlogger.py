from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict

from CampusCompass.app.config import CHATS_DIR


@dataclass
class _Session:
    path: Path
    started_at: datetime
    last_seen: datetime
    lock: Lock


class ChatFileLogger:
    """
    1 logfile per 'chat session' (per sender_id, met idle-timeout).
    Bestandsnaam: chat_YYYYMMDD_HHMM.txt
    """
    def __init__(self, *, idle_new_file_minutes: int = 60) -> None:
        self._idle = timedelta(minutes=idle_new_file_minutes)
        self._sessions: Dict[str, _Session] = {}
        self._global_lock = Lock()

    def _now(self) -> datetime:
        return datetime.now()

    def _new_path(self, now: datetime) -> Path:
        date = now.strftime("%Y%m%d")
        start = now.strftime("%H%M%S")
        base = CHATS_DIR / f"chat_{date}_{start}.txt"

        i = 1
        path = base
        while path.exists():
            path = CHATS_DIR / f"chat_{date}_{start}_{i}.txt"
            i += 1
        return path

    def _get_session(self, sender_id: str) -> _Session:
        now = self._now()
        with self._global_lock:
            sess = self._sessions.get(sender_id)

            if sess and (now - sess.last_seen) <= self._idle:
                sess.last_seen = now
                return sess

            sess = _Session(
                path=self._new_path(now),
                started_at=now,
                last_seen=now,
                lock=Lock(),
            )
            self._sessions[sender_id] = sess
            return sess

    def append(self, sender_id: str, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        sess = self._get_session(sender_id)
        timestamp = self._now().strftime("%H_%M_%S")
        block = f"{speaker}\n{text} | {timestamp}\n\n"

        with sess.lock:
            sess.path.parent.mkdir(parents=True, exist_ok=True)
            with sess.path.open("a", encoding="utf-8") as f:
                f.write(block)


LOGGER = ChatFileLogger(idle_new_file_minutes=60)