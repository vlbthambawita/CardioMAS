from __future__ import annotations

import uuid

from cardiomas.schemas.memory import SessionMemory
from cardiomas.schemas.tools import ToolCallRecord


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def start(self) -> SessionMemory:
        session = SessionMemory(session_id=str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionMemory:
        return self._sessions[session_id]

    def append_query(self, session_id: str, query: str) -> None:
        self._sessions[session_id].queries.append(query)

    def append_tool_call(self, session_id: str, call: ToolCallRecord) -> None:
        self._sessions[session_id].tool_history.append(call)
