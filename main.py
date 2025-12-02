# main.py
# -*- coding: utf-8 -*-

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from report_generator import generate_match_report

# ログ設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ロガーのインスタンスを作成
logger = logging.getLogger(__name__)

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
    try:
        logger.debug(f"Received payload: {payload}")  # ログで受け取ったデータを表示
        payload_dict = payload.dict()  # Pydanticモデル → dict に変換してそのまま渡す
        report = generate_match_report(payload_dict)  # レポート生成関数の呼び出し
        logger.info("レポート生成完了")  # 成功時のログ
        return ReportResponse(report=report)
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")  # エラーが発生した場合
        raise HTTPException(status_code=500, detail="レポート生成中にエラーが発生しました")
