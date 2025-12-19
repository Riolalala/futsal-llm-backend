# report_generator.py
# -*- coding: utf-8 -*-

import os
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

# ===================== Logging =====================
logger = logging.getLogger("report_generator")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI()

# ===================== Config =====================
SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", "https://futsal-report-api.onrender.com").rstrip("/")

MODEL_NAME = os.getenv("LLM_MODEL", "o4-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # recommended by OpenAI docs :contentReference[oaicite:9]{index=9}
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "512"))      # can reduce memory if needed :contentReference[oaicite:10]{index=10}
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

MAX_IMAGES = int(os.getenv("MAX_IMAGES", "12"))  # safety: avoid huge payloads

SKIP_SNAPSHOT_TYPES = {
    "交代", "たいむあうと", "タイムアウト",
    "substitution", "timeout",
}

# ===================== System Prompt =====================
def load_system_prompt() -> str:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        sp = config.get("system_prompt") or ""
        if not sp.strip():
            raise ValueError("system_prompt is empty")
        logger.info("[PROMPT] loaded config.json system_prompt length=%d", len(sp))
        return sp
    except Exception as e:
        logger.error("[PROMPT] failed to load config.json: %s", e)
        # fallback prompt (minimal)
        return (
            "あなたはフットサルの試合レポートを書くスポーツライターです。"
            "与えられた試合記録と戦術ボード画像をもとに、日本語で読みやすいレポートを書いてください。"
        )

SYSTEM_PROMPT = load_system_prompt()

# ===================== Knowledge Base (KB) =====================
# ここはあなたの futsal_knowledge を「ID付き」で保持（ログにIDと中身を出すため）
KB: List[Dict[str, str]] = [
    {"id": "KB001", "text": "フットサルは通常、2×20分のハーフで行われ、インターバルは10分である。"},
    {"id": "KB002", "text": "フットサルではサッカーと異なりオフサイドがない。"},
    {"id": "KB003", "text": "フットサルでは1チーム5人（GK1人＋FP4人）で構成される。"},
    {"id": "KB004", "text": "フットサルの試合でファウルが累積され、5回を超えると相手にフリーキックが与えられる。"},
    {"id": "KB005", "text": "3-1フォーメーション（低め）は3人が守備、1人（ピヴォ）が攻撃の起点となる。"},
    {"id": "KB006", "text": "2-2フォーメーション（ボックス型）は2人守備＋2人攻撃が近接し、切替を素早く行う。"},
    {"id": "KB007", "text": "4-0（クワトロ）は4人が横並びでパス回しし、相手を動かしてズレを作る。"},
    {"id": "KB008", "text": "カットインはサイドから中央へ切れ込み、シュートやラストパスを狙う動き。"},
    {"id": "KB009", "text": "キックインはサイドラインを割った際に足で再開するセットプレー。"},
    {"id": "KB010", "text": "スクリーンは味方が相手DFの進路を遮り、パスコースやシュートコースを作る。"},
    {"id": "KB011", "text": "デスマルケはマークを外して受ける動きで、パスの受け所やシュート機会を作る。"},
    {"id": "KB012", "text": "パワープレーはGKをフィールドに上げ、数的優位で攻撃を強化する。"},
    {"id": "KB013", "text": "ハーフは自陣でコンパクトに守り、縦パスを制限しつつ奪って速攻を狙う守り方。"},
    {"id": "KB014", "text": "セットプレー（FK/CK）ではショートやダイレクトを使い分け、ズレを作る。"},
]

KB_TEXTS = [x["text"] for x in KB]
KB_IDS = [x["id"] for x in KB]

# lazy cache
_KB_EMB: Optional[np.ndarray] = None  # shape: (N, D) normalized
_KB_READY: bool = False

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom

def _get_embeddings(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings create :contentReference[oaicite:11]{index=11}
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[t.replace("\n", " ") for t in texts],
        dimensions=EMBED_DIMENSIONS,
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def _ensure_kb_embeddings() -> None:
    global _KB_EMB, _KB_READY
    if _KB_READY and _KB_EMB is not None:
        return
    logger.info("[RAG] building KB embeddings: N=%d model=%s dim=%d", len(KB_TEXTS), EMBED_MODEL, EMBED_DIMENSIONS)
    vecs = _get_embeddings(KB_TEXTS)
    _KB_EMB = _l2_normalize(vecs)
    _KB_READY = True
    logger.info("[RAG] KB embeddings ready")

def rag_search(query: str, k: int = 4) -> List[Tuple[str, float, str]]:
    _ensure_kb_embeddings()
    assert _KB_EMB is not None
    qv = _get_embeddings([query])
    qv = _l2_normalize(qv)[0]  # (D,)
    scores = _KB_EMB @ qv  # cosine similarity
    idx = np.argsort(scores)[::-1][:k]
    out: List[Tuple[str, float, str]] = []
    for i in idx:
        out.append((KB_IDS[i], float(scores[i]), KB_TEXTS[i]))
    return out

# ===================== Helpers =====================
def normalize_half(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    sl = s.lower()

    # first half patterns
    if sl in {"1st", "1h", "h1", "first", "firsthalf", "first_half", "1", "前半", "前"} or "1st" in sl or "first" in sl or "前半" in s:
        return "1st"
    # second half patterns
    if sl in {"2nd", "2h", "h2", "second", "secondhalf", "second_half", "2", "後半", "後"} or "2nd" in sl or "second" in sl or "後半" in s:
        return "2nd"
    return None

def is_skip_snapshot_event(ev_type: str) -> bool:
    if not ev_type:
        return False
    t = ev_type.strip().lower()
    if t in SKIP_SNAPSHOT_TYPES:
        return True
    # Japanese contains check
    if "交代" in ev_type or "タイムアウト" in ev_type:
        return True
    return False

def to_image_url(snapshot_path: Any) -> Optional[str]:
    if not snapshot_path:
        return None
    if not isinstance(snapshot_path, str):
        return None
    sp = snapshot_path.strip()
    if not sp or sp == "string":
        return None

    # ✅ data URL is allowed (do NOT skip)
    if sp.startswith("data:image/"):
        return sp

    if sp.startswith("http://") or sp.startswith("https://"):
        return sp

    # relative path -> our server
    if not sp.startswith("/"):
        sp = "/" + sp
    return SNAPSHOT_BASE_URL + sp

def build_event_text(ev: Dict[str, Any]) -> str:
    half = normalize_half(ev.get("half")) or "不明"
    minute_raw = ev.get("minute")
    second_raw = ev.get("second")
    team_side = ev.get("teamSide")
    ev_type = ev.get("type") or ""
    note = ev.get("note") or ""
    main_no = ev.get("mainPlayerNumber")
    assist_no = ev.get("assistPlayerNumber")

    # time string
    time_str = half
    try:
        if minute_raw is not None and second_raw is not None:
            m = int(minute_raw)
            s = int(second_raw)
            time_str = f"{half} {m:02d}:{s:02d}"
    except Exception:
        pass

    if team_side == "home":
        side_str = "ホーム"
    elif team_side == "away":
        side_str = "アウェイ"
    else:
        side_str = "チーム不明"

    players = []
    if main_no is not None:
        players.append(f"主:#{main_no}")
    if assist_no is not None:
        players.append(f"補:#{assist_no}")

    pstr = (" " + " ".join(players)) if players else ""
    nstr = f" メモ:{note}" if note else ""
    return f"[{time_str}] {side_str} {ev_type}{pstr}{nstr}".strip()

def extract_used_kb_ids(text: str) -> List[str]:
    """
    model output tail: <RAG_USED>KB001,KB003</RAG_USED>
    """
    m = re.search(r"<RAG_USED>\s*([^<]+?)\s*</RAG_USED>", text, flags=re.IGNORECASE)
    if not m:
        return []
    inside = m.group(1).strip()
    if not inside or inside.lower() in {"none", "null"}:
        return []
    ids = [x.strip() for x in inside.split(",") if x.strip()]
    # keep only existing KB ids
    kbset = set(KB_IDS)
    return [i for i in ids if i in kbset]

def strip_rag_used_tag(text: str) -> str:
    return re.sub(r"\s*<RAG_USED>.*?</RAG_USED>\s*", "\n", text, flags=re.IGNORECASE | re.DOTALL).strip()

# ===================== Message Builder =====================
def build_messages(match_payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float, str]]]:
    match_id = match_payload.get("matchId")
    venue = match_payload.get("venue") or "会場不明"
    tournament = match_payload.get("tournament") or "大会名不明"
    round_desc = match_payload.get("round") or "ラウンド不明"
    kickoff = match_payload.get("kickoffISO8601") or "日時不明"

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = home.get("name", "ホーム")
    away_name = away.get("name", "アウェイ")

    events = match_payload.get("events", []) or []

    # count halves & snapshots
    half_norm = [normalize_half(ev.get("half")) for ev in events]
    n1 = sum(1 for h in half_norm if h == "1st")
    n2 = sum(1 for h in half_norm if h == "2nd")
    nU = len(events) - n1 - n2

    snap_count = 0
    for ev in events:
        sp = ev.get("snapshotPath")
        if sp and isinstance(sp, str) and sp.strip() and sp.strip() != "string":
            snap_count += 1

    logger.info("[PAYLOAD] events=%d (1st=%d, 2nd=%d, unknown=%d) snapshots=%d venue=%s",
                len(events), n1, n2, nU, snap_count, venue)

    # Build query for RAG
    # ここは試合全体に合わせてクエリを作る（例：出てくるフォーメーション/セットプレー/守備など）
    query = f"{tournament} {home_name} vs {away_name} の試合レポート。フォーメーション、守備、セットプレーの観点。"
    rag_hits = rag_search(query, k=RAG_TOP_K)

    logger.info("[RAG] top=%d", len(rag_hits))
    for kid, score, text in rag_hits:
        logger.info("[RAG] %s score=%.3f text=%s", kid, score, text)

    # Build user content
    user_content: List[Dict[str, Any]] = []

    header_text = (
        f"【試合概要】\n"
        f"大会: {tournament}\n"
        f"ラウンド: {round_desc}\n"
        f"会場: {venue}\n"
        f"日時(キックオフ想定): {kickoff}\n"
        f"対戦カード: {home_name} vs {away_name}\n"
        f"matchId: {match_id}\n"
        f"前半イベント数: {n1} / 後半イベント数: {n2} / half不明: {nU}\n"
        f"※ 後半イベントが 0 件の場合、後半を創作しないこと。\n"
    )
    user_content.append({"type": "input_text", "text": header_text})

    rag_block = "【参照可能な知識（RAG）】\n" + "\n".join([f"- [{kid}] {text}" for kid, _, text in rag_hits])
    user_content.append({"type": "input_text", "text": rag_block})

    intro = (
        "以下にイベント一覧を示します。各イベントには（あれば）戦術ボード画像が続きます。\n"
        "画像が無いイベントはテキストのみで判断してください。\n"
    )
    user_content.append({"type": "input_text", "text": intro})

    # group events
    first_half = [ev for ev in events if normalize_half(ev.get("half")) == "1st"]
    second_half = [ev for ev in events if normalize_half(ev.get("half")) == "2nd"]
    unknown_half = [ev for ev in events if normalize_half(ev.get("half")) is None]

    img_used = 0

    def add_events(title: str, evs: List[Dict[str, Any]]):
        nonlocal img_used
        user_content.append({"type": "input_text", "text": f"\n--- {title}（{len(evs)}件）---\n"})
        for idx, ev in enumerate(evs, start=1):
            ev_text = build_event_text(ev)
            user_content.append({"type": "input_text", "text": f"{idx}. {ev_text}"})

            ev_type = (ev.get("type") or "").strip()
            sp = ev.get("snapshotPath")

            if is_skip_snapshot_event(ev_type):
                continue

            if img_used >= MAX_IMAGES:
                continue

            url = to_image_url(sp)
            if url:
                user_content.append({"type": "input_image", "image_url": url})
                img_used += 1

    add_events("前半", first_half)

    if second_half:
        add_events("後半", second_half)
    else:
        user_content.append({"type": "input_text", "text": "\n--- 後半（0件：未提供）---\n"})

    if unknown_half:
        # 後半と混同されやすいので、明確に「half不明」と書く
        add_events("half不明（要修正）", unknown_half)

    logger.info("[SEND] messages=2 images_attached=%d", img_used)

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    return messages, rag_hits

# ===================== Main API =====================
def generate_match_report(match_payload: Dict[str, Any]) -> str:
    messages, _rag_hits = build_messages(match_payload)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=messages,
    )

    out = resp.output_text or ""
    used_ids = extract_used_kb_ids(out)

    logger.info("[RESULT] length=%d", len(out))

    if used_ids:
        logger.info("[RAG-USED] ids=%s", ",".join(used_ids))
        kb_map = {x["id"]: x["text"] for x in KB}
        for kid in used_ids:
            logger.info("[RAG-USED] %s text=%s", kid, kb_map.get(kid, ""))
    else:
        logger.info("[RAG-USED] ids=(none)")

    # remove internal tag before returning to app
    out_clean = strip_rag_used_tag(out)
    return out_clean
