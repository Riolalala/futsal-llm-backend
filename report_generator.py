# report_generator.py
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, DefaultDict
from collections import defaultdict
from pathlib import Path

import httpx
from openai import OpenAI
from openai import BadRequestError

# ===== optional: matplotlib (png export only) =====
HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# =========================
# Logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("report_generator")

OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "180"))  # 例: 180秒
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

client = OpenAI(
    timeout=httpx.Timeout(
        connect=10.0,
        read=OPENAI_TIMEOUT_SEC,
        write=OPENAI_TIMEOUT_SEC,
        pool=OPENAI_TIMEOUT_SEC,
    ),
    max_retries=OPENAI_MAX_RETRIES,
)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://futsal-report-api.onrender.com").rstrip("/")
SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", PUBLIC_BASE_URL).rstrip("/")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
REPORT_MODEL = os.getenv("REPORT_MODEL", "gpt-4o")

# ✅ 評価用: 既定を 0 に（環境変数で上書き可能）
REPORT_TEMPERATURE = float(os.getenv("REPORT_TEMPERATURE", "1"))

SNAPSHOT_CHECK_TIMEOUT = float(os.getenv("SNAPSHOT_CHECK_TIMEOUT", "3.0"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "24"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))

# =========================
# Evaluation condition (C0/C1/C2)
# =========================
# ✅ コメントアウトで切り替え（評価時に使う条件だけ残す）
REPORT_CONDITION = "C2"
# REPORT_CONDITION = "C1"
# REPORT_CONDITION = "C0"
#
# C0：RAGなし／簡易プロンプト（ベースライン）
# C1：RAGなし／プロンプト改善のみ
# C2：RAGあり＋プロンプト改善（提案法）

# charts
ENABLE_CHARTS = os.getenv("ENABLE_CHARTS", "1").lower() not in ("0", "false", "no")
CHART_TOPK = int(os.getenv("CHART_TOPK", "8"))

SKIP_SNAPSHOT_TYPES = {
    "substitution", "change", "swap", "交代",
    "timeout", "time_out", "timeOut", "タイムアウト",
}

logger.info("[CHARTS] ENABLE_CHARTS=%s HAVE_MPL=%s", ENABLE_CHARTS, HAVE_MPL)
logger.info("[MODEL] REPORT_MODEL=%s REPORT_TEMPERATURE=%s", REPORT_MODEL, REPORT_TEMPERATURE)
logger.info("[COND] REPORT_CONDITION=%s", REPORT_CONDITION)

# =========================
# Import prompt / knowledge
# =========================
try:
    from .prompt_config import system_prompt as SYSTEM_PROMPT_BASE
    from .prompt_config import safe_append as SYSTEM_PROMPT_SAFE_APPEND
    from .futsal_knowledge import futsal_knowledge
except Exception:
    from prompt_config import system_prompt as SYSTEM_PROMPT_BASE
    from prompt_config import safe_append as SYSTEM_PROMPT_SAFE_APPEND
    from futsal_knowledge import futsal_knowledge

SYSTEM_PROMPT = (SYSTEM_PROMPT_BASE or "").strip() or "あなたはフットサルの試合レポートを書くスポーツライターです。"
SYSTEM_PROMPT_SAFE_APPEND = (SYSTEM_PROMPT_SAFE_APPEND or "").strip()

# ✅ 出力タグを強制（RAG_USED抽出が安定）
OUTPUT_FORMAT_APPEND = """
【出力形式（必須）】
- 返答は必ず次の2ブロックのみで出力してください（順番固定）。
<report_md>
（ここに監督・選手向けのMarkdownレポート本文だけ）
</report_md>
<rag_used>
（本文で参考にしたKBのIDをカンマ区切り。使っていない場合は none）
</rag_used>
"""
SYSTEM_PROMPT = SYSTEM_PROMPT + "\n" + OUTPUT_FORMAT_APPEND
if SYSTEM_PROMPT_SAFE_APPEND:
    SYSTEM_PROMPT = SYSTEM_PROMPT + "\n" + SYSTEM_PROMPT_SAFE_APPEND

logger.info("[PROMPT] loaded from prompt_config.py (len=%d)", len(SYSTEM_PROMPT))

# =========================
# Knowledge Base (KBxxx)
# =========================
KB_ITEMS: List[Dict[str, str]] = [
    {"id": f"KB{idx:03d}", "text": txt} for idx, txt in enumerate(futsal_knowledge, start=1)
]

_kb_lock = threading.Lock()
_kb_ready = False
_kb_vecs: List[List[float]] = []

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
        score = _dot(qvec, kvec)
        scored.append((item["id"], score, item["text"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# =========================
# Event helpers
# =========================
_HALF_RE_1 = re.compile(r"(?:^|\b)(1|1st|1h|first)(?:\b|$)", re.IGNORECASE)
_HALF_RE_2 = re.compile(r"(?:^|\b)(2|2nd|2h|second)(?:\b|$)", re.IGNORECASE)

def normalize_half(raw: Any) -> Optional[str]:
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

    if "前半" in s or s_lower in ("前",):
        return "1st"
    if "後半" in s or s_lower in ("後",):
        return "2nd"

    if _HALF_RE_1.search(s_lower) or "firsthalf" in s_lower or "first_half" in s_lower:
        return "1st"
    if _HALF_RE_2.search(s_lower) or "secondhalf" in s_lower or "second_half" in s_lower:
        return "2nd"

    if s_lower in ("1", "1st", "1h", "first", "firsthalf", "first_half"):
        return "1st"
    if s_lower in ("2", "2nd", "2h", "second", "secondhalf", "second_half"):
        return "2nd"

    return s

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

_url_ok_cache: Dict[str, Tuple[bool, float]] = {}

def url_is_ok(url: str) -> bool:
    now = time.time()
    cached = _url_ok_cache.get(url)
    if cached:
        ok, ts = cached
        ttl = 300.0 if ok else 30.0
        if now - ts < ttl:
            return ok

    headers = {"Range": "bytes=0-0"}
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
        try:
            ok = _try(timeout_slow)
        except Exception as e2:
            logger.info("[URL_CHECK_ERR] url=%s err=%r (slow)", url, e2)
            ok = False

    _url_ok_cache[url] = (ok, now)
    return ok

def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def _team_name(team_side: Any, home_name: str, away_name: str) -> str:
    if team_side == "home":
        return home_name
    if team_side == "away":
        return away_name
    return "チーム不明"

def _player_name(team_side: Any, number: Optional[int], home_map: Dict[int, str], away_map: Dict[int, str]) -> str:
    if number is None:
        return ""
    if team_side == "home":
        return home_map.get(number, "")
    if team_side == "away":
        return away_map.get(number, "")
    return home_map.get(number, "") or away_map.get(number, "")

def _is_goal(ev_type: str) -> bool:
    s = (ev_type or "").lower()
    return ("goal" in s) or ("ゴール" in ev_type) or ("得点" in ev_type)

def _is_shot(ev_type: str) -> bool:
    s = (ev_type or "").lower()
    return ("shot" in s) or ("シュート" in ev_type)

def _is_foul(ev_type: str) -> bool:
    s = (ev_type or "").lower()
    return ("foul" in s) or ("ファウル" in ev_type)

def _is_sub(ev_type: str) -> bool:
    s = (ev_type or "").lower()
    return ("substitution" in s) or ("交代" in ev_type)

def _is_timeout(ev_type: str) -> bool:
    s = (ev_type or "").lower()
    return ("timeout" in s) or ("タイムアウト" in ev_type)

def _build_event_text(
    ev: Dict[str, Any],
    home_name: str,
    away_name: str,
    home_map: Dict[int, str],
    away_map: Dict[int, str],
) -> str:
    h = normalize_half(ev.get("half"))
    if h == "1st":
        half_label = "前半"
    elif h == "2nd":
        half_label = "後半"
    else:
        half_label = "half不明（要修正）"

    minute = _safe_int(ev.get("minute"))
    second = _safe_int(ev.get("second"))

    team_side = ev.get("teamSide")
    team_str = _team_name(team_side, home_name, away_name)

    main_no = _safe_int(ev.get("mainPlayerNumber"))
    assist_no = _safe_int(ev.get("assistPlayerNumber"))
    note = (ev.get("note") or "").strip()
    ev_type = (ev.get("type") or "").strip()

    if minute is not None and second is not None:
        time_str = f"{half_label} {minute:02d}:{second:02d}"
    elif minute is not None:
        time_str = f"{half_label} {minute:02d}:??"
    else:
        time_str = f"{half_label} 時間不明"

    main_name = _player_name(team_side, main_no, home_map, away_map)
    assist_name = _player_name(team_side, assist_no, home_map, away_map)

    def fmt(no: Optional[int], name: str) -> str:
        if no is None:
            return ""
        return f"#{no} {name}".strip()

    players_str = ""
    if _is_sub(ev_type):
        out_s = fmt(main_no, main_name)
        in_s = fmt(assist_no, assist_name)
        parts = []
        if out_s:
            parts.append(f"OUT {out_s}")
        if in_s:
            parts.append(f"IN {in_s}")
        players_str = " / ".join(parts)
    elif _is_goal(ev_type):
        scorer = fmt(main_no, main_name)
        assist = fmt(assist_no, assist_name)
        if scorer and assist:
            players_str = f"得点: {scorer} / A: {assist}"
        elif scorer:
            players_str = f"得点: {scorer}"
        elif assist:
            players_str = f"A: {assist}"
    else:
        p = fmt(main_no, main_name)
        a = fmt(assist_no, assist_name)
        if p and a:
            players_str = f"関与: {p} / {a}"
        elif p:
            players_str = f"関与: {p}"
        elif a:
            players_str = f"関与: {a}"

    note_str = f" メモ: {note}" if note else ""
    body = f"[{time_str}] {team_str} の {ev_type}。"
    if players_str:
        body += f" {players_str}。"
    if note_str:
        body += note_str
    return body.strip()

def event_priority(ev: Dict[str, Any]) -> int:
    t = (ev.get("type") or "").lower()
    if "goal" in t or "ゴール" in (ev.get("type") or ""):
        return 100
    if "shot" in t or "シュート" in (ev.get("type") or ""):
        return 80
    if "foul" in t or "ファウル" in (ev.get("type") or ""):
        return 60
    if "kick" in t or "キックイン" in (ev.get("type") or "") or "corner" in t or "コーナー" in (ev.get("type") or ""):
        return 50
    return 10

# =========================
# STATS builder
# =========================
def _roster_map(team: Dict[str, Any]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    players = team.get("players") or []
    if not isinstance(players, list):
        return m
    for p in players:
        try:
            no = _safe_int(p.get("number"))
            name = (p.get("name") or "").strip()
            if no is not None:
                m[no] = name
        except Exception:
            continue
    return m

def _time_sec(minute: Optional[int], second: Optional[int]) -> Optional[int]:
    if minute is None or second is None:
        return None
    return int(minute) * 60 + int(second)

def build_stats(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = (home.get("name") or "HOME").strip()
    away_name = (away.get("name") or "AWAY").strip()

    home_map = _roster_map(home)
    away_map = _roster_map(away)

    events = match_payload.get("events", []) or []

    half_event_counts = {"first": 0, "second": 0, "unknown": 0}

    cats = ["goals", "shots", "fouls", "subs", "timeouts", "others"]
    def init_counter() -> Dict[str, int]:
        return {c: 0 for c in cats}

    team_half_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        "first": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
        "second": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
        "unknown": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
    }

    player_stats: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    goals_timeline: List[Dict[str, Any]] = []
    misc_timeline: List[Dict[str, Any]] = []

    def half_bucket(ev: Dict[str, Any]) -> str:
        h = normalize_half(ev.get("half"))
        if h == "1st":
            return "first"
        if h == "2nd":
            return "second"
        return "unknown"

    def pkey(team_side_: Any, no: Optional[int]) -> Optional[str]:
        if no is None:
            return None
        team_label = _team_name(team_side_, home_name, away_name)
        pname = _player_name(team_side_, no, home_map, away_map)
        return f"{team_label}|#{no} {pname}".strip()

    for ev in events:
        hb = half_bucket(ev)
        half_event_counts[hb] += 1

        team_side = ev.get("teamSide")
        tname = _team_name(team_side, home_name, away_name)

        ev_type = (ev.get("type") or "").strip()
        minute = _safe_int(ev.get("minute"))
        second = _safe_int(ev.get("second"))
        ts = _time_sec(minute, second)

        main_no = _safe_int(ev.get("mainPlayerNumber"))
        assist_no = _safe_int(ev.get("assistPlayerNumber"))

        if _is_goal(ev_type):
            team_half_counts[hb][tname]["goals"] += 1
        elif _is_shot(ev_type):
            team_half_counts[hb][tname]["shots"] += 1
        elif _is_foul(ev_type):
            team_half_counts[hb][tname]["fouls"] += 1
        elif _is_sub(ev_type):
            team_half_counts[hb][tname]["subs"] += 1
        elif _is_timeout(ev_type):
            team_half_counts[hb][tname]["timeouts"] += 1
        else:
            team_half_counts[hb][tname]["others"] += 1

        if _is_goal(ev_type):
            if tname != "チーム不明":
                scorer_k = pkey(team_side, main_no)
                assist_k = pkey(team_side, assist_no)
                if scorer_k:
                    player_stats[scorer_k]["goals"] += 1
                if assist_k:
                    player_stats[assist_k]["assists"] += 1

                goals_timeline.append({
                    "half": hb,
                    "minute": minute, "second": second, "time_sec": ts,
                    "team": tname,
                    "scorer_no": main_no,
                    "scorer_name": _player_name(team_side, main_no, home_map, away_map),
                    "assist_no": assist_no,
                    "assist_name": _player_name(team_side, assist_no, home_map, away_map),
                    "note": ev.get("note") or "",
                })
            else:
                misc_timeline.append({
                    "half": hb, "minute": minute, "second": second, "time_sec": ts,
                    "team": tname,
                    "type": ev_type,
                    "note": "チーム不明ゴール（スコア計算外）",
                })
        else:
            if _is_shot(ev_type):
                k = pkey(team_side, main_no)
                if k:
                    player_stats[k]["shots"] += 1
            if _is_foul(ev_type):
                k = pkey(team_side, main_no)
                if k:
                    player_stats[k]["fouls"] += 1
            if _is_sub(ev_type):
                out_k = pkey(team_side, main_no)
                in_k = pkey(team_side, assist_no)
                if out_k:
                    player_stats[out_k]["sub_out"] += 1
                if in_k:
                    player_stats[in_k]["sub_in"] += 1

            misc_timeline.append({
                "half": hb, "minute": minute, "second": second, "time_sec": ts,
                "team": tname,
                "type": ev_type,
                "main_no": main_no,
                "assist_no": assist_no,
                "note": ev.get("note") or "",
            })

    def score(hb: str) -> Tuple[int, int]:
        hg = sum(1 for g in goals_timeline if g["half"] == hb and g["team"] == home_name)
        ag = sum(1 for g in goals_timeline if g["half"] == hb and g["team"] == away_name)
        return hg, ag

    h1, a1 = score("first")
    h2, a2 = score("second")
    ht, at = h1 + h2, a1 + a2

    rows: List[Dict[str, Any]] = []
    for k, d in player_stats.items():
        team, player = k.split("|", 1) if "|" in k else ("チーム不明", k)
        rows.append({
            "team": team,
            "player": player.strip(),
            "goals": int(d.get("goals", 0)),
            "assists": int(d.get("assists", 0)),
            "shots": int(d.get("shots", 0)),
            "fouls": int(d.get("fouls", 0)),
            "sub_in": int(d.get("sub_in", 0)),
            "sub_out": int(d.get("sub_out", 0)),
        })
    rows.sort(key=lambda r: (r["goals"] + r["assists"], r["shots"]), reverse=True)

    goals_timeline.sort(key=lambda x: (x["half"], x["time_sec"] if x["time_sec"] is not None else 10**9))
    misc_timeline.sort(key=lambda x: (x["half"], x["time_sec"] if x["time_sec"] is not None else 10**9))

    return {
        "match": {
            "matchId": match_payload.get("matchId") or "",
            "home_team": home_name,
            "away_team": away_name,
            "tournament": match_payload.get("tournament") or "",
            "round": match_payload.get("round") or "",
            "venue": match_payload.get("venue") or "",
            "kickoffISO8601": match_payload.get("kickoffISO8601") or "",
        },
        "half_event_counts": half_event_counts,
        "score": {
            "first": {"home": h1, "away": a1},
            "second": {"home": h2, "away": a2},
            "total": {"home": ht, "away": at},
        },
        "team_half_counts": team_half_counts,
        "goals_timeline": goals_timeline,
        "misc_timeline": misc_timeline,
        "player_stats": rows,
    }

# =========================
# Charts (JSON always, PNG optional)
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_fig_png(fig, save_path: Path) -> None:
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def _chart_url(image_path: str) -> str:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    if not image_path.startswith("/"):
        image_path = "/" + image_path
    return PUBLIC_BASE_URL + image_path

def build_charts(stats: Dict[str, Any], report_dir: Optional[str]) -> List[Dict[str, Any]]:
    """
    ✅ 重要: matplotlib が無くても “data(JSON)” は返す（iOS描画用）
    png は HAVE_MPL かつ report_dir があるときだけ作る
    """
    if not ENABLE_CHARTS:
        return []

    m = stats["match"]
    match_id = (m.get("matchId") or "no_match_id").strip()
    home = m["home_team"]
    away = m["away_team"]

    out: List[Dict[str, Any]] = []
    can_png = bool(report_dir) and HAVE_MPL

    def sum_half(h: str, team: str, key: str) -> int:
        return int(stats["team_half_counts"][h][team].get(key, 0))

    # 1) Team summary (bar)
    keys = ["goals", "shots", "fouls", "subs", "timeouts"]
    labels = ["Goals", "Shots", "Fouls", "Subs", "Timeouts"]
    home_vals = [sum_half("first", home, k) + sum_half("second", home, k) for k in keys]
    away_vals = [sum_half("first", away, k) + sum_half("second", away, k) for k in keys]

    chart = {
        "id": "team_summary",
        "title": "チーム比較（主要イベント数）",
        "kind": "bar_team_events",
        "imagePath": None,
        "data": {
            "labels": labels,
            "home": {"team": home, "values": home_vals},
            "away": {"team": away, "values": away_vals},
        }
    }

    if can_png:
        try:
            fig = plt.figure(figsize=(10.5, 5.5))
            x = list(range(len(labels)))
            plt.bar([i - 0.2 for i in x], home_vals, width=0.4, label=home)
            plt.bar([i + 0.2 for i in x], away_vals, width=0.4, label=away)
            plt.xticks(x, labels)
            plt.title("チーム比較（主要イベント数）")
            plt.legend()
            save_path = Path(report_dir) / match_id / "team_summary.png"
            _save_fig_png(fig, save_path)
            chart["imagePath"] = f"/reports/{match_id}/team_summary.png"
        except Exception as e:
            logger.exception("[CHARTS] failed to save team_summary png: %r", e)

    out.append(chart)

    # 2) Score timeline (step)
    goals = stats.get("goals_timeline") or []

    def to_minsec(hb: str, minute: Optional[int], second: Optional[int]) -> int:
        base = 0
        if hb == "second":
            base = 20 * 60
        elif hb == "unknown":
            base = 40 * 60
        if minute is None or second is None:
            return base
        return base + int(minute) * 60 + int(second)

    points = []
    h_sc = 0
    a_sc = 0
    for g in goals:
        t = to_minsec(g["half"], g.get("minute"), g.get("second"))
        if g["team"] == home:
            h_sc += 1
        elif g["team"] == away:
            a_sc += 1
        points.append({"t": t, "home": h_sc, "away": a_sc, "team": g["team"]})

    chart2 = {
        "id": "score_timeline",
        "title": "得点推移（累積）",
        "kind": "step_score_timeline",
        "imagePath": None,
        "data": {
            "homeTeam": home,
            "awayTeam": away,
            "points": points
        }
    }

    if can_png:
        try:
            fig = plt.figure(figsize=(10.5, 4.8))
            if points:
                xs = [p["t"] for p in points]
                hs = [p["home"] for p in points]
                as_ = [p["away"] for p in points]
                plt.step(xs, hs, where="post", label=home)
                plt.step(xs, as_, where="post", label=away)
            plt.title("得点推移（累積）")
            plt.xlabel("時間（前半0-20分 / 後半20-40分）")
            plt.ylabel("得点（累積）")
            plt.legend()
            save_path = Path(report_dir) / match_id / "score_timeline.png"
            _save_fig_png(fig, save_path)
            chart2["imagePath"] = f"/reports/{match_id}/score_timeline.png"
        except Exception as e:
            logger.exception("[CHARTS] failed to save score_timeline png: %r", e)

    out.append(chart2)

    # 3) Top player shots per team (bar)
    def player_top(team: str, key: str, topk: int) -> Tuple[List[str], List[int]]:
        rows = [r for r in (stats.get("player_stats") or []) if r["team"] == team]
        rows.sort(key=lambda r: r.get(key, 0), reverse=True)
        rows = rows[:topk]
        return [r["player"] for r in rows], [int(r.get(key, 0)) for r in rows]

    for team in (home, away):
        names, vals = player_top(team, "shots", CHART_TOPK)
        if not names:
            continue
        cid = f"top_shots_{'home' if team==home else 'away'}"
        c = {
            "id": cid,
            "title": f"シュート数 上位（{team}）",
            "kind": "bar_player_top_shots",
            "imagePath": None,
            "data": {"team": team, "labels": names, "values": vals}
        }
        if can_png:
            try:
                fig = plt.figure(figsize=(11.0, 5.2))
                plt.bar(names, vals)
                plt.xticks(rotation=30, ha="right")
                plt.title(f"シュート数 上位{min(CHART_TOPK, len(names))}（{team}）")
                plt.tight_layout()
                save_path = Path(report_dir) / match_id / f"{cid}.png"
                _save_fig_png(fig, save_path)
                c["imagePath"] = f"/reports/{match_id}/{cid}.png"
            except Exception as e:
                logger.exception("[CHARTS] failed to save %s png: %r", cid, e)
        out.append(c)

    # 4) NEW: shots per minute (line)  ※iOSでLineMark
    events = stats.get("misc_timeline") or []
    series = []  # [{team, minute, shots}]

    def add_shot(team: str, hb: str, minute: Optional[int]) -> int:
        if minute is None:
            return -1
        base = 0
        if hb == "second":
            base = 20
        elif hb == "unknown":
            base = 40
        return base + int(minute)

    cnt: Dict[Tuple[str, int], int] = defaultdict(int)
    for ev in events:
        if not _is_shot(ev.get("type") or ""):
            continue
        team = ev.get("team") or "チーム不明"
        hb = ev.get("half")
        minute = ev.get("minute")
        tmin = add_shot(team, hb, minute)
        if tmin >= 0:
            cnt[(team, tmin)] += 1

    for team in (home, away):
        for tmin in range(0, 41):
            series.append({"team": team, "minute": tmin, "shots": int(cnt.get((team, tmin), 0))})

    out.append({
        "id": "shots_momentum",
        "title": "シュート推移（1分ごと）",
        "kind": "line_shots_per_minute",
        "imagePath": None,
        "data": {
            "homeTeam": home,
            "awayTeam": away,
            "points": series
        }
    })

    # 5) NEW: event share (pie) ※iOSでSectorMark
    def total(team: str, key: str) -> int:
        return (sum_half("first", team, key) + sum_half("second", team, key))

    share_labels = ["Goals", "Shots", "Fouls", "Subs", "Timeouts", "Others"]
    totals = {
        "Goals": total(home, "goals") + total(away, "goals"),
        "Shots": total(home, "shots") + total(away, "shots"),
        "Fouls": total(home, "fouls") + total(away, "fouls"),
        "Subs": total(home, "subs") + total(away, "subs"),
        "Timeouts": total(home, "timeouts") + total(away, "timeouts"),
        "Others": total(home, "others") + total(away, "others"),
    }
    pie_items = [{"label": k, "value": int(totals[k])} for k in share_labels]

    out.append({
        "id": "event_share",
        "title": "イベント構成比（全体）",
        "kind": "pie_event_share",
        "imagePath": None,
        "data": {"items": pie_items}
    })

    return out

# =========================
# Markdown builders (tables)
# =========================
def stats_tables_markdown(stats: Dict[str, Any]) -> str:
    """
    ✅ 本文に不要な段落（イベント件数/個人指標など）を出さない版。
    ここではスコア表だけ出す。
    """
    m = stats["match"]
    home = m["home_team"]
    away = m["away_team"]
    sc = stats["score"]

    lines: List[str] = []
    lines.append("# 試合分析レポート\n")
    lines.append("## スコア（集計はコード確定）\n")
    lines.append("| | 前半 | 後半 | 合計 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| {home} | {sc['first']['home']} | {sc['second']['home']} | {sc['total']['home']} |")
    lines.append(f"| {away} | {sc['first']['away']} | {sc['second']['away']} | {sc['total']['away']} |")
    return "\n".join(lines) + "\n"

def charts_markdown(charts: List[Dict[str, Any]]) -> str:
    """
    ⚠️ 互換のため関数は残すが、本文(full_md)には連結しない。
    """
    items = [c for c in charts if c.get("imagePath")]
    if not items:
        return ""
    md = ["\n\n## グラフ（自動生成）\n"]
    for c in items:
        title = c.get("title", "")
        url = _chart_url(c["imagePath"])
        md.append(f"### {title}\n\n![]({url})\n")
    return "\n".join(md)

# =========================
# Build messages (LLM input)
# =========================
def build_messages(
    match_payload: Dict[str, Any],
    rag_hits: List[Tuple[str, float, str]],
    stats_json: Dict[str, Any],
    allow_images: bool = True,
    include_rag: bool = True,                     # ✅ 追加：RAGブロックを入れるか
    system_prompt_override: Optional[str] = None  # ✅ 追加：C0用にsystem promptを差し替え
) -> List[Dict[str, Any]]:
    venue = match_payload.get("venue") or "会場不明"
    tournament = match_payload.get("tournament") or "大会名不明"
    round_desc = match_payload.get("round") or "ラウンド不明"
    kickoff = match_payload.get("kickoffISO8601") or "日時不明"

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = (home.get("name") or "HOME").strip()
    away_name = (away.get("name") or "AWAY").strip()

    home_map = _roster_map(home)
    away_map = _roster_map(away)

    header_text = (
        f"大会: {tournament}\n"
        f"ラウンド: {round_desc}\n"
        f"会場: {venue}\n"
        f"日時(キックオフ想定): {kickoff}\n"
        f"対戦カード: {home_name} vs {away_name}\n"
    )

    events = match_payload.get("events", []) or []

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

    snapshot_candidates = [ev for ev in events if should_attach_snapshot(ev)]
    logger.info(
        "[PAYLOAD] events=%d (1st=%d, 2nd=%d, unknown=%d) snapshots=%d venue=%s",
        len(events), len(first_half), len(second_half), len(unknown_half), len(snapshot_candidates), venue
    )

    user_content: List[Dict[str, Any]] = []

    intro_text = (
        "以下にフットサルの試合記録と、各イベントに対応する戦術ボード画像（ある場合）を与えます。\n"
        "テキスト情報（時間・チーム名・選手番号/名前・メモなど）と画像の両方を踏まえて、"
        "systemメッセージの指示に従い、監督・選手向けの技術/戦術レポート（Markdown）を書いてください。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
        "各イベントは「前半/後半/half不明」に分けて提示します。\n"
    )
    user_content.append({"type": "input_text", "text": intro_text})

    stats_text = json.dumps(stats_json, ensure_ascii=False, indent=2)
    user_content.append({
        "type": "input_text",
        "text": "【STATS_JSON（重要：この中が事実。スコア/得点/集計は改変・再計算禁止）】\n```json\n" + stats_text + "\n```\n"
    })

    # ✅ C2のみRAGブロックを投入（C0/C1は完全に入れない）
    if include_rag and rag_hits:
        rag_text_lines = []
        for kid, score, ktext in rag_hits:
            rag_text_lines.append(f"{kid} score={score:.3f} text={ktext}")
        user_content.append({
            "type": "input_text",
            "text": "【参考知識（RAG）】\n" + "\n".join(rag_text_lines) + "\n"
        })

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

    if first_half:
        user_content.append({"type": "input_text", "text": "\n--- 前半 (1st) ---\n"})
        for i, ev in enumerate(first_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[1st-{i}] {_build_event_text(ev, home_name, away_name, home_map, away_map)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})

    if second_half:
        user_content.append({"type": "input_text", "text": "\n--- 後半 (2nd) ---\n"})
        for i, ev in enumerate(second_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[2nd-{i}] {_build_event_text(ev, home_name, away_name, home_map, away_map)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})
    else:
        user_content.append({"type": "input_text", "text": "\n【後半イベント】未提供（2ndは0件）。後半について推測で書かないでください。\n"})

    if unknown_half:
        user_content.append({"type": "input_text", "text": "\n--- half不明 (unknown) ---\n"})
        user_content.append({"type": "input_text", "text": "※ これらは前半/後半が不明です。後半扱いにせず『half不明（要修正）』として扱ってください。\n"})
        for i, ev in enumerate(unknown_half, start=1):
            user_content.append({"type": "input_text", "text": f"\n[UNK-{i}] {_build_event_text(ev, home_name, away_name, home_map, away_map)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})

    sys_prompt = (system_prompt_override or SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
        {"role": "user", "content": user_content},
    ]
    logger.info("[SEND] allow_images=%s chosen_images=%d include_rag=%s", allow_images, len(chosen_images), include_rag)
    return messages

# =========================
# Output extraction
# =========================
_RAG_USED_RE = re.compile(r"<\s*rag_used\s*>(.*?)</\s*rag_used\s*>", re.DOTALL | re.IGNORECASE)
_REPORT_MD_RE = re.compile(r"<\s*report_md\s*>(.*?)</\s*report_md\s*>", re.DOTALL | re.IGNORECASE)

def extract_rag_used_ids(text: str) -> List[str]:
    m = _RAG_USED_RE.search(text or "")
    if not m:
        ids = sorted(set(re.findall(r"KB\d{3}", text or "")))
        return ids
    inside = m.group(1).strip()
    if inside.lower() in ("none", "ids=none") or inside == "":
        return []
    inside = inside.replace("\n", " ").replace(" ", ",")
    ids = [x.strip() for x in inside.split(",") if x.strip()]
    return [x for x in ids if re.fullmatch(r"KB\d{3}", x)]

def strip_tag_block(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = rf"\s*<\s*{tag}\s*>.*?</\s*{tag}\s*>\s*"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_report_md(text: str) -> str:
    m = _REPORT_MD_RE.search(text or "")
    if m:
        return (m.group(1) or "").strip()
    return (text or "").strip()

def strip_rag_used_footer(text: str) -> str:
    return strip_tag_block(text, "rag_used")

# ✅ 保険：LLMが本文に「グラフ/チャート…」段落を混ぜても強制削除
_GRAPH_SECTION_RE = re.compile(
    r"""
    (?:\n|^)                 # 行頭
    \#{2,3}\s*               # ## または ###
    .*?(グラフ|チャート|可視化|自動生成|プロット|図表|表示|画面).*
    \n
    (.*?)(?=\n\#{2}\s|\Z)    # 次の "## " または EOF まで削除
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE
)

def strip_graph_sections(md: str) -> str:
    if not md:
        return ""
    out = re.sub(_GRAPH_SECTION_RE, "\n", md)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

# =========================
# OpenAI call helper (temperature fallback)
# =========================
def _responses_create_with_fallback(model: str, input_messages: List[Dict[str, Any]], temperature: float) -> str:
    kwargs: Dict[str, Any] = {"model": model, "input": input_messages}
    kwargs["temperature"] = temperature
    try:
        resp = client.responses.create(**kwargs)
        return resp.output_text or ""
    except BadRequestError as e:
        msg = str(e)
        if "Unsupported parameter" in msg and "temperature" in msg:
            logger.warning("[OPENAI] temperature unsupported for model=%s -> retry without temperature", model)
            kwargs.pop("temperature", None)
            resp2 = client.responses.create(**kwargs)
            return resp2.output_text or ""
        raise

# =========================
# Main API function
# =========================
def generate_match_report_bundle(match_payload: Dict[str, Any], report_dir: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()

    stats = build_stats(match_payload)
    charts = build_charts(stats, report_dir=report_dir)

    venue = match_payload.get("venue") or ""
    tournament = match_payload.get("tournament") or ""
    events = match_payload.get("events", []) or []

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = (home.get("name") or "HOME").strip()
    away_name = (away.get("name") or "AWAY").strip()
    home_map = _roster_map(home)
    away_map = _roster_map(away)

    # ===== condition switch =====
    cond = (REPORT_CONDITION or "C2").strip().upper()
    use_rag = (cond == "C2")            # ✅ C2のみRAGあり
    use_simple_prompt = (cond == "C0")  # ✅ C0のみ簡易プロンプト（system差替え）

    # RAG query（C2のときだけ使う）
    sample_events_text = "\n".join(
        _build_event_text(ev, home_name, away_name, home_map, away_map) for ev in events[:30]
    )
    rag_query = f"{tournament} {venue}\n{sample_events_text}\n戦術 フォーメーション セットプレイ 守備 攻撃"

    rag_hits: List[Tuple[str, float, str]] = []
    if use_rag:
        rag_hits = rag_search(rag_query, top_k=RAG_TOP_K)
        logger.info("[RAG] cond=%s top=%d", cond, len(rag_hits))
    else:
        logger.info("[RAG] cond=%s disabled", cond)

    # system prompt の切替（C0だけ簡易）
    system_prompt_for_run = SYSTEM_PROMPT
    if use_simple_prompt:
        system_prompt_for_run = (
            "あなたはフットサルの試合レポートを書くアシスタントです。\n"
            "入力の STATS_JSON を最優先の根拠として、事実を改変せずに簡潔にまとめてください。\n"
            "推測や未記録の作り話はしないでください。\n"
        ) + OUTPUT_FORMAT_APPEND + ("\n" + SYSTEM_PROMPT_SAFE_APPEND if SYSTEM_PROMPT_SAFE_APPEND else "")

    messages = build_messages(
        match_payload,
        rag_hits,
        stats_json=stats,
        allow_images=True,
        include_rag=use_rag,
        system_prompt_override=system_prompt_for_run
    )

    try:
        out = _responses_create_with_fallback(REPORT_MODEL, messages, REPORT_TEMPERATURE)
    except BadRequestError as e:
        msg = str(e)
        if "Timeout while downloading" in msg or ("param" in msg and "url" in msg):
            logger.warning("[OPENAI] image download failed -> retry without images: %s", msg)
            messages2 = build_messages(
                match_payload,
                rag_hits,
                stats_json=stats,
                allow_images=False,
                include_rag=use_rag,
                system_prompt_override=system_prompt_for_run
            )
            out = _responses_create_with_fallback(REPORT_MODEL, messages2, REPORT_TEMPERATURE)
        else:
            logger.exception("[OPENAI] BadRequestError: %s", msg)
            raise

    tail = (out or "")[-300:].replace("\n", "\\n")
    logger.info("[RAW_TAIL] %s", tail)

    used_ids = extract_rag_used_ids(out)
    logger.info("[RESULT] cond=%s length=%d sec=%.2f", cond, len(out), time.time() - t0)
    logger.info("[RAG-USED] ids=%s", ",".join(used_ids) if used_ids else "none")

    report_md = extract_report_md(out)
    report_md = strip_rag_used_footer(report_md)

    # ✅ 念のため：本文内に「グラフ/チャート…」段落があれば削除
    report_md = strip_graph_sections(report_md)

    header_md = stats_tables_markdown(stats)

    # ✅ 重要：グラフは別画面で表示するので本文に混ぜない
    full_md = header_md + "\n" + report_md

    return {
        "condition": cond,          # ✅ 評価で便利（C0/C1/C2）
        "temperature": REPORT_TEMPERATURE,
        "report_md": full_md,
        "rag_used_ids": used_ids,
        "stats": stats,
        "charts": charts,          # ← グラフ用データは返す
        "has_charts": bool(charts),
    }

def generate_match_report(match_payload: Dict[str, Any]) -> str:
    bundle = generate_match_report_bundle(match_payload, report_dir=None)
    return bundle["report_md"]
