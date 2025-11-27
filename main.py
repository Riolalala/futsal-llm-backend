# main.py
# -*- coding: utf-8 -*-

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from report_generator import generate_match_report


# ==== Swift側の LLMPayload に対応する Pydantic モデル ====

class LLMPlayer(BaseModel):
    number: int
    name: str
    position: str


class LLMTeamInfo(BaseModel):
    name: str
    players: List[LLMPlayer]


class LLMEvent(BaseModel):
    id: str
    type: str
    half: Optional[str] = None
    minute: Optional[int] = None
    second: Optional[int] = None
    teamSide: Optional[str] = None
    mainPlayerNumber: Optional[int] = None
    assistPlayerNumber: Optional[int] = None
    note: Optional[str] = None
    snapshotPath: Optional[str] = None


class LLMPayload(BaseModel):
    matchId: str
    venue: Optional[str] = None
    tournament: str
    round: str
    kickoffISO8601: str
    home: LLMTeamInfo
    away: LLMTeamInfo
    events: List[LLMEvent]


class ReportResponse(BaseModel):
    report: str


# ==== FastAPI アプリ ====

app = FastAPI()


@app.post("/generate_report", response_model=ReportResponse)
def generate_report(payload: LLMPayload):
    """
    iOS アプリから LLMPayload を受け取り、
    o4-mini で試合レポートを生成して返すエンドポイント。
    """
    # Pydanticモデル → dict に変換してそのまま渡す
    report = generate_match_report(payload.model_dump())
    return ReportResponse(report=report)
