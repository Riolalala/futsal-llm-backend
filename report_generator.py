# report_generator.py
# -*- coding: utf-8 -*-

import os
import re
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple

import httpx
from openai import OpenAI
from openai import BadRequestError

# =========================
# Logging (Render logs に出る)
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("report_generator")

client = OpenAI()

SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", "https://futsal-report-api.onrender.com").rstrip("/")

# OpenAI Embedding
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# レポート生成モデル
REPORT_MODEL = os.getenv("REPORT_MODEL", "gpt-4o")
#LLM Temperature
REPORT_TEMPERATURE = float(os.getenv("REPORT_TEMPERATURE", "0"))

# 画像の疎通確認タイムアウト（秒）
SNAPSHOT_CHECK_TIMEOUT = float(os.getenv("SNAPSHOT_CHECK_TIMEOUT", "3.0"))
# 1回の生成で送る最大画像数（多すぎると失敗しやすい）
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "24"))
# RAGで入れる知識数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# 「このtypeは画像いらない」方針（必要なら追加）
SKIP_SNAPSHOT_TYPES = {
    "substitution", "change", "swap", "交代",
    "timeout", "time_out", "timeOut", "タイムアウト",
}

# =========================
# Import prompt / knowledge (py files)
# =========================
try:
    # package 配下でも動くように相対importを優先
    from .prompt_config import system_prompt as SYSTEM_PROMPT_BASE
    from .prompt_config import safe_append as SYSTEM_PROMPT_SAFE_APPEND
    from .futsal_knowledge import futsal_knowledge
except Exception:
    from prompt_config import system_prompt as SYSTEM_PROMPT_BASE
    from prompt_config import safe_append as SYSTEM_PROMPT_SAFE_APPEND
    from futsal_knowledge import futsal_knowledge

SYSTEM_PROMPT = (SYSTEM_PROMPT_BASE or "").strip() or "あなたはフットサルの試合レポートを書くスポーツライターです。"
SYSTEM_PROMPT_SAFE_APPEND = (SYSTEM_PROMPT_SAFE_APPEND or "").strip()

logger.info("[PROMPT] loaded from prompt_config.py (len=%d)", len(SYSTEM_PROMPT))
logger.info("[PROMPT_HEAD] %s", SYSTEM_PROMPT.replace("\n", " ")[:180])

# =========================
# Knowledge Base (KBxxx)
# =========================
KB_ITEMS: List[Dict[str, str]] = [
    {"id": f"KB{idx:03d}", "text": txt} for idx, txt in enumerate(futsal_knowledge, start=1)
]

# KB embedding cache
_kb_lock = threading.Lock()
_kb_ready = False
_kb_vecs: List[List[float]] = []  # normalized embeddings

def _l2norm(vec: List[float]) -> List[float]:
    s = 0.0
    for x in vec:
        s += x * x
    if s <= 0:
        return vec
    inv = 1.0 / (s ** 0.5)
    return [x * inv for x in vec]

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ensure_kb_index() -> None:
    global _kb_ready, _kb_vecs
    if _kb_ready:
        return
    with _kb_lock:
        if _kb_ready:
            return
        logger.info("[RAG] building KB embeddings... items=%d model=%s", len(KB_ITEMS), EMBED_MODEL)
        vecs = _embed_texts([k["text"] for k in KB_ITEMS])
        _kb_vecs = [_l2norm(v) for v in vecs]
        _kb_ready = True
        logger.info("[RAG] KB embeddings ready.")

def rag_search(query: str, top_k: int = 4) -> List[Tuple[str, float, str]]:
    ensure_kb_index()
    qvec = _l2norm(_embed_texts([query])[0])

    scored: List[Tuple[str, float, str]] = []
    for item, kvec in zip(KB_ITEMS, _kb_vecs):
        score = _dot(qvec, kvec)  # cosine similarity
        scored.append((item["id"], score, item["text"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# =========================
# Event helpers
# =========================
def normalize_half(raw: Any) -> Optional[str]:
    """
    返り値は "1st"/"2nd"/None/その他
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if int(raw) == 1:
            return "1st"
        if int(raw) == 2:
            return "2nd"
        return str(raw)

    s = str(raw).strip()
    if not s:
        return None

    s_lower = s.lower()
    if s in ("1st", "1h", "前半", "前", "first", "firsthalf") or s_lower in ("1st", "1h", "first", "firsthalf"):
        return "1st"
    if s in ("2nd", "2h", "後半", "後", "second", "secondhalf") or s_lower in ("2nd", "2h", "second", "secondhalf"):
        return "2nd"

    return s  # unknown扱い

def should_attach_snapshot(ev: Dict[str, Any]) -> bool:
    ev_type = (ev.get("type") or "").strip()
    if ev_type in SKIP_SNAPSHOT_TYPES:
        return False
    sp = ev.get("snapshotPath")
    if not sp or str(sp).strip() == "" or sp == "string":
        return False
    if str(sp).startswith("data:"):
        return False
    return True

def resolve_snapshot_url(snapshot_path: str) -> str:
    if snapshot_path.startswith("http://") or snapshot_path.startswith("https://"):
        return snapshot_path
    if not snapshot_path.startswith("/"):
        snapshot_path = "/" + snapshot_path
    return SNAPSHOT_BASE_URL + snapshot_path

# URLチェックの簡易キャッシュ（短時間の同一URLチェックを減らす）
_url_ok_cache: Dict[str, Tuple[bool, float]] = {}  # url -> (ok, timestamp)

def url_is_ok(url: str) -> bool:
    now = time.time()
    cached = _url_ok_cache.get(url)
    if cached:
        ok, ts = cached
        # OKは5分、NGは30秒だけキャッシュ
        ttl = 300.0 if ok else 30.0
        if now - ts < ttl:
            return ok

    headers = {"Range": "bytes=0-0"}  # 1byteだけ
    timeout_fast = httpx.Timeout(connect=SNAPSHOT_CHECK_TIMEOUT, read=SNAPSHOT_CHECK_TIMEOUT, write=SNAPSHOT_CHECK_TIMEOUT, pool=SNAPSHOT_CHECK_TIMEOUT)
    timeout_slow = httpx.Timeout(connect=max(5.0, SNAPSHOT_CHECK_TIMEOUT), read=max(10.0, SNAPSHOT_CHECK_TIMEOUT), write=max(10.0, SNAPSHOT_CHECK_TIMEOUT), pool=max(10.0, SNAPSHOT_CHECK_TIMEOUT))

    def _try(timeout: httpx.Timeout) -> bool:
        r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        ok_status = r.status_code in (200, 206)
        ok_type = ("image/" in ct) or url.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        if not (ok_status and ok_type):
            logger.info("[URL_CHECK_NG] status=%s ct=%s url=%s", r.status_code, ct, url)
            return False
        return True

    try:
        ok = _try(timeout_fast)
    except Exception as e:
        logger.info("[URL_CHECK_ERR] url=%s err=%r (fast)", url, e)
        # Renderのコールドスタート等を想定し、1回だけ長めでリトライ
        try:
            ok = _try(timeout_slow)
        except Exception as e2:
            logger.info("[URL_CHECK_ERR] url=%s err=%r (slow)", url, e2)
            ok = False

    _url_ok_cache[url] = (ok, now)
    return ok

def _build_event_text(ev: Dict[str, Any]) -> str:
    h = normalize_half(ev.get("half"))
    if h == "1st":
        half_label = "前半"
    elif h == "2nd":
        half_label = "後半"
    else:
        half_label = "half不明（要修正）"

    minute_raw = ev.get("minute")
    second_raw = ev.get("second")

    team_side = ev.get("teamSide")
    main_no = ev.get("mainPlayerNumber")
    assist_no = ev.get("assistPlayerNumber")
    note = ev.get("note") or ""
    ev_type = ev.get("type") or ""

    minute = None
    second = None
    try:
        if minute_raw is not None:
            minute = int(minute_raw)
        if second_raw is not None:
            second = int(second_raw)
    except (TypeError, ValueError):
        minute = None
        second = None

    if minute is not None and second is not None:
        time_str = f"{half_label} {minute:02d}:{second:02d}"
    elif minute is not None:
        time_str = f"{half_label} {minute:02d}:??"
    else:
        time_str = f"{half_label} 時間不明"

    if team_side == "home":
        side_str = "ホームチーム"
    elif team_side == "away":
        side_str = "アウェイチーム"
    else:
        side_str = "チーム不明"

    players_str = ""
    if main_no is not None:
        players_str += f" 主な関与選手: 背番号{main_no}"
    if assist_no is not None:
        players_str += f"、アシスト: 背番号{assist_no}" if players_str else f" アシスト: 背番号{assist_no}"

    note_str = f" メモ: {note}" if note else ""
    return f"[{time_str}] {side_str} の {ev_type}。{players_str}{note_str}".strip()

def event_priority(ev: Dict[str, Any]) -> int:
    """
    画像を送る優先度（大きいほど優先）
    """
    t = (ev.get("type") or "").lower()
    if "goal" in t or "ゴール" in t:
        return 100
    if "shot" in t or "シュート" in t:
        return 80
    if "foul" in t or "ファウル" in t:
        return 60
    if "kick" in t or "キックイン" in t or "corner" in t or "コーナー" in t:
        return 50
    return 10

# =========================
# Build messages
# =========================
def build_messages(
    match_payload: Dict[str, Any],
    rag_hits: List[Tuple[str, float, str]],
    allow_images: bool = True
) -> List[Dict[str, Any]]:
    venue = match_payload.get("venue") or "会場不明"
    tournament = match_payload.get("tournament") or "大会名不明"
    round_desc = match_payload.get("round") or "ラウンド不明"
    kickoff = match_payload.get("kickoffISO8601") or "日時不明"

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = home.get("name", "ホーム")
    away_name = away.get("name", "アウェイ")

    header_text = (
        f"大会: {tournament}\n"
        f"ラウンド: {round_desc}\n"
        f"会場: {venue}\n"
        f"日時(キックオフ想定): {kickoff}\n"
        f"対戦カード: {home_name} vs {away_name}\n"
    )

    events = match_payload.get("events", []) or []

    # half 正規化して分類
    first_half: List[Dict[str, Any]] = []
    second_half: List[Dict[str, Any]] = []
    unknown_half: List[Dict[str, Any]] = []

    for ev in events:
        h = normalize_half(ev.get("half"))
        if h == "1st":
            first_half.append(ev)
        elif h == "2nd":
            second_half.append(ev)
        else:
            unknown_half.append(ev)

    # snapshot 数（送れる候補）
    snapshot_candidates = [ev for ev in events if should_attach_snapshot(ev)]
    logger.info(
        "[PAYLOAD] events=%d (1st=%d, 2nd=%d, unknown=%d) snapshots=%d venue=%s",
        len(events), len(first_half), len(second_half), len(unknown_half), len(snapshot_candidates), venue
    )

    user_content: List[Dict[str, Any]] = []

    intro_text = (
        "以下にフットサルの試合記録と、各イベントに対応する戦術ボード画像（ある場合）を与えます。\n"
        "テキスト情報（時間・チーム・選手番号・メモなど）と画像の両方を踏まえて、"
        "systemメッセージの指示に従い、試合レポートを書いてください。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
        "各イベントは「前半/後半/half不明」に分けて提示します。\n"
    )
    user_content.append({"type": "input_text", "text": intro_text})

    # RAG を user に注入
    rag_text_lines = []
    for kid, score, ktext in rag_hits:
        rag_text_lines.append(f"{kid} score={score:.3f} text={ktext}")
    user_content.append({
        "type": "input_text",
        "text": "【参考知識（RAG）】\n" + "\n".join(rag_text_lines) + "\n"
    })

    # 画像の送信数を絞る（優先度順）
    chosen_images: List[Dict[str, Any]] = []
    if allow_images and snapshot_candidates:
        snapshot_candidates.sort(key=event_priority, reverse=True)
        for ev in snapshot_candidates:
            if len(chosen_images) >= MAX_IMAGES:
                break
            sp = str(ev.get("snapshotPath"))
            url = resolve_snapshot_url(sp)
            if url_is_ok(url):
                chosen_images.append({"ev": ev, "url": url})
            else:
                logger.info("[SNAPSHOT_SKIP] not reachable url=%s", url)

    # 前半
    if first_half:
        user_content.append({"type": "input_text", "text": "\n--- 前半 (1st) ---\n"})
        for i, ev in enumerate(first_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[1st-{i}] {_build_event_text(ev)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})

    # 後半
    if second_half:
        user_content.append({"type": "input_text", "text": "\n--- 後半 (2nd) ---\n"})
        for i, ev in enumerate(second_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[2nd-{i}] {_build_event_text(ev)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})
    else:
        # 後半創作防止（user側にも明示）
        user_content.append({"type": "input_text", "text": "\n【後半イベント】未提供（2ndは0件）。後半について推測で書かないでください。\n"})

    # half不明
    if unknown_half:
        user_content.append({"type": "input_text", "text": "\n--- half不明 (unknown) ---\n"})
        user_content.append({"type": "input_text", "text": "※ これらは前半/後半が不明です。後半扱いにせず『half不明（要修正）』として扱ってください。\n"})
        for i, ev in enumerate(unknown_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[UNK-{i}] {_build_event_text(ev)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": SYSTEM_PROMPT + ("\n\n" + SYSTEM_PROMPT_SAFE_APPEND if SYSTEM_PROMPT_SAFE_APPEND else "")}
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    logger.info("[SEND] messages=%d allow_images=%s chosen_images=%d", len(messages), allow_images, len(chosen_images))
    return messages

# =========================
# RAG used extraction
# =========================
_RAG_USED_RE = re.compile(r"<\s*rag_used\s*>(.*?)</\s*rag_used\s*>", re.DOTALL | re.IGNORECASE)

def extract_rag_used_ids(text: str) -> List[str]:
    m = _RAG_USED_RE.search(text or "")
    if not m:
        ids = sorted(set(re.findall(r"KB\d{3}", text or "")))
        return ids
    inside = m.group(1).strip()
    if inside.lower() == "none" or inside == "":
        return []
    ids = [x.strip() for x in inside.split(",") if x.strip()]
    return [x for x in ids if re.fullmatch(r"KB\d{3}", x)]

def strip_rag_used_footer(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s*<\s*rag_used\s*>.*?</\s*rag_used\s*>\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

# =========================
# Main API function
# =========================
def generate_match_report(match_payload: Dict[str, Any]) -> str:
    """
    FastAPI から呼び出される想定。
    画像URLが原因でOpenAIが400を返したら、画像なしでリトライして落ちないようにする。
    """
    t0 = time.time()

    venue = match_payload.get("venue") or ""
    tournament = match_payload.get("tournament") or ""
    events = match_payload.get("events", []) or []
    sample_events_text = "\n".join(_build_event_text(ev) for ev in events[:30])  # 先頭30だけ
    rag_query = f"{tournament} {venue}\n{sample_events_text}\n戦術 フォーメーション セットプレイ 守備 攻撃"

    rag_hits = rag_search(rag_query, top_k=RAG_TOP_K)
    logger.info("[RAG] top=%d", len(rag_hits))
    for kid, score, ktext in rag_hits:
        logger.info("[RAG] %s score=%.3f text=%s", kid, score, (ktext[:90] + "…") if len(ktext) > 90 else ktext)

    # 1st try: allow images
    messages = build_messages(match_payload, rag_hits, allow_images=True)

    try:
        resp = client.responses.create(
            model=REPORT_MODEL,
            input=messages,
            temperature=REPORT_TEMPERATURE,
        )
        out = resp.output_text or ""
    except BadRequestError as e:
        msg = str(e)
        if "Timeout while downloading" in msg or ("param" in msg and "url" in msg):
            logger.warning("[OPENAI] image download failed -> retry without images: %s", msg)
            messages2 = build_messages(match_payload, rag_hits, allow_images=False)
            resp2 = client.responses.create(
                model=REPORT_MODEL,
                input=messages2,
                temperature=REPORT_TEMPERATURE,
            )
            out = resp2.output_text or ""
        else:
            logger.exception("[OPENAI] BadRequestError (not image-url): %s", msg)
            raise

    tail = (out or "")[-300:].replace("\n", "\\n")
    logger.info("[RAW_TAIL] %s", tail)

    used_ids = extract_rag_used_ids(out)
    logger.info("[RESULT] length=%d sec=%.2f", len(out), time.time() - t0)
    logger.info("[RAG-USED] ids=%s", ",".join(used_ids) if used_ids else "none")

    used_set = set(used_ids)
    for item in KB_ITEMS:
        if item["id"] in used_set:
            logger.info("[RAG-USED] %s text=%s", item["id"], (item["text"][:120] + "…") if len(item["text"]) > 120 else item["text"])

    # ユーザーに返す本文からは <RAG_USED> を消す
    return strip_rag_used_footer(out)
