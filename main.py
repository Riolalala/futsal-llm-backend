# main.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path

from report_generator import generate_match_report_bundle

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

class ReportChart(BaseModel):
    id: str
    title: str
    kind: str
    imagePath: Optional[str] = None
    data: Dict[str, Any]

class ReportResponse(BaseModel):
    reportMd: str
    ragUsedIds: List[str]
    stats: Dict[str, Any]
    charts: List[ReportChart]
    hasCharts: bool

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASE_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

@app.post("/upload_snapshot")
async def upload_snapshot(
    matchId: str = Form(...),
    eventId: str = Form(...),
    file: UploadFile = File(...),
):
    logger.debug(
        "upload_snapshot: matchId=%s, eventId=%s, filename=%s content_type=%s",
        matchId, eventId, file.filename, file.content_type,
    )

    match_dir = SNAPSHOT_DIR / matchId
    match_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{eventId}.png"
    save_path = match_dir / filename

    content = await file.read()
    save_path.write_bytes(content)

    snapshot_path = f"/snapshots/{matchId}/{filename}"
    logger.debug("saved snapshot to %s (bytes=%d), snapshotPath=%s", save_path, len(content), snapshot_path)

    return JSONResponse({"snapshotPath": snapshot_path})

@app.post("/generate_report", response_model=ReportResponse)
async def generate_report_endpoint(payload: LLMPayload):
    logger.debug("generate_report called")

    try:
        match_dict = payload.model_dump()

        evs = match_dict.get("events", []) or []
        with_sp = sum(1 for e in evs if e.get("snapshotPath"))
        logger.info("[PAYLOAD] events=%d with_snapshotPath=%d venue=%s",
                    len(evs), with_sp, match_dict.get("venue"))

        bundle = await asyncio.to_thread(
            generate_match_report_bundle,
            match_dict,
            str(REPORT_DIR),
        )

        charts = bundle.get("charts", []) or []
        return ReportResponse(
            reportMd=bundle["report_md"],
            ragUsedIds=bundle.get("rag_used_ids", []),
            stats=bundle["stats"],
            charts=charts,
            hasCharts=bool(charts),
        )

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"レポート生成中にエラーが発生しました: {repr(e)}",
        )
