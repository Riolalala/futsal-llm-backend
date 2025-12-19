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
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

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
        "upload_snapshot: matchId=%s, eventId=%s, filename=%s",
        matchId, eventId, file.filename,
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
    logger.debug("saved snapshot to %s, snapshotPath=%s", save_path, snapshot_path)

    return JSONResponse({"snapshotPath": snapshot_path})

# ========== 試合レポート生成エンドポイント ==========

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report_endpoint(payload: LLMPayload):
    """
    iOS から LLMPayload（試合記録＋イベント情報）が送られてくる想定。
    report_generator.generate_match_report() を呼び出して、
    日本語の試合レポートを返す。
    """
    logger.debug("generate_report called")
    try:
        # Pydantic モデル → dict に変換して LLM へ
        match_dict = payload.model_dump()
        logger.debug("payload (dict) = %s", match_dict)

        # レポート生成
        report = generate_match_report(match_dict)

        logger.debug("report generated successfully, length=%d", len(report))
        return ReportResponse(report=report)

    except Exception as e:
        # ★ ここでサーバ側ログにフルのトレースバックを出す
        logger.exception("エラーが発生しました: %s", e)

        # ★ そしてクライアント側にもエラー内容をそのまま返す（デバッグ用）
        raise HTTPException(
            status_code=500,
            detail=f"レポート生成中にエラーが発生しました: {repr(e)}",
        )
