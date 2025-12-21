import asyncio
import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from report_generator import generate_match_report
from typing import List, Optional
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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

# スナップショット保存ディレクトリ
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ★これがないと /snapshots/... が404になりやすい
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")

# ========== 画像アップロード用エンドポイント ==========

@app.post("/upload_snapshot")
async def upload_snapshot(
    matchId: str = Form(...),
    eventId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    iOS から送られてきた戦術ボード画像を保存し、
    クライアントが使う snapshotPath（相対パス）を返す。
    """
    logger.debug(
        "upload_snapshot: matchId=%s, eventId=%s, filename=%s content_type=%s",
        matchId, eventId, file.filename, file.content_type,
    )

    # 保存先ディレクトリ: snapshots/<matchId>/
    match_dir = SNAPSHOT_DIR / matchId
    match_dir.mkdir(parents=True, exist_ok=True)

    # 拡張子は一旦 .png に固定（必要なら file.filename から推定してもOK）
    filename = f"{eventId}.png"
    save_path = match_dir / filename

    content = await file.read()
    save_path.write_bytes(content)

    # クライアントに返す snapshotPath（LLMPayload.events[].snapshotPath に入れる）
    snapshot_path = f"/snapshots/{matchId}/{filename}"
    logger.debug("saved snapshot to %s (bytes=%d), snapshotPath=%s", save_path, len(content), snapshot_path)

    return JSONResponse({"snapshotPath": snapshot_path})

# ========== 試合レポート生成エンドポイント ==========

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report_endpoint(payload: LLMPayload):
    logger.debug("generate_report called")

    try:
        match_dict = payload.model_dump()

        evs = match_dict.get("events", []) or []
        with_sp = sum(1 for e in evs if e.get("snapshotPath"))
        logger.info("[PAYLOAD] events=%d with_snapshotPath=%d venue=%s",
                    len(evs), with_sp, match_dict.get("venue"))

        # ✅ ここが重要：同期の重い処理をイベントループ上で回さない
        report = await asyncio.to_thread(generate_match_report, match_dict)

        logger.debug("report generated successfully, length=%d", len(report))
        return ReportResponse(report=report)

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"レポート生成中にエラーが発生しました: {repr(e)}",
        )
