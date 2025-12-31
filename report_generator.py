# report_generator.py
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import base64
import io
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, DefaultDict
from collections import defaultdict

import httpx
from openai import OpenAI
from openai import BadRequestError

# ===== optional: matplotlib (charts) =====
HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

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
# LLM Temperature
REPORT_TEMPERATURE = float(os.getenv("REPORT_TEMPERATURE", "0"))

# 画像の疎通確認タイムアウト（秒）
SNAPSHOT_CHECK_TIMEOUT = float(os.getenv("SNAPSHOT_CHECK_TIMEOUT", "3.0"))
# 1回の生成で送る最大画像数（多すぎると失敗しやすい）
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "24"))
# RAGで入れる知識数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))

# charts
ENABLE_CHARTS = os.getenv("ENABLE_CHARTS", "1").lower() not in ("0", "false", "no")
CHART_TOPK = int(os.getenv("CHART_TOPK", "8"))

# 「このtypeは画像いらない」方針（必要なら追加）
SKIP_SNAPSHOT_TYPES = {
    "substitution", "change", "swap", "交代",
    "timeout", "time_out", "timeOut", "タイムアウト",
}

# =========================
# Import prompt / knowledge (py files)
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

    # number
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

    # string number
    if s_lower in ("1", "1st", "1h", "first", "firsthalf", "前半", "前"):
        return "1st"
    if s_lower in ("2", "2nd", "2h", "second", "secondhalf", "後半", "後"):
        return "2nd"

    # extra forms
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
    # team不明なら両方から探す
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

    # player strings with names
    main_name = _player_name(team_side, main_no, home_map, away_map)
    assist_name = _player_name(team_side, assist_no, home_map, away_map)

    def fmt(no: Optional[int], name: str) -> str:
        if no is None:
            return ""
        return f"#{no} {name}".strip()

    players_str = ""
    if _is_sub(ev_type):
        # OUT=main / IN=assist の想定
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
        # shot/foul etc
        p = fmt(main_no, main_name)
        a = fmt(assist_no, assist_name)
        if p and a:
            players_str = f"関与: {p} / {a}"
        elif p:
            players_str = f"関与: {p}"
        elif a:
            players_str = f"関与: {a}"

    note_str = f" メモ: {note}" if note else ""
    # ✅ “ホームチーム/アウェイチーム”は禁止 → チーム名
    body = f"[{time_str}] {team_str} の {ev_type}。"
    if players_str:
        body += f" {players_str}。"
    if note_str:
        body += note_str
    return body.strip()

def event_priority(ev: Dict[str, Any]) -> int:
    """
    画像を送る優先度（大きいほど優先）
    """
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
# STATS builder (facts are computed here; LLM must not recalc)
# =========================
def _roster_map(team: Dict[str, Any]) -> Dict[int, str]:
    """
    payload の players から {number: name} を作る（無くても落ちない）
    想定: players=[{"number": 10, "name": "xxx"}, ...]
    """
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

    # team/half counters for key categories
    cats = ["goals", "shots", "fouls", "subs", "timeouts", "others"]
    def init_counter() -> Dict[str, int]:
        return {c: 0 for c in cats}

    team_half_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        "first": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
        "second": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
        "unknown": {home_name: init_counter(), away_name: init_counter(), "チーム不明": init_counter()},
    }

    # player stats
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

        # categorize
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

        # player label
        def pkey(team_side_: Any, no: Optional[int]) -> Optional[str]:
            if no is None:
                return None
            team_label = _team_name(team_side_, home_name, away_name)
            pname = _player_name(team_side_, no, home_map, away_map)
            return f"{team_label}|#{no} {pname}".strip()

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

    # score from known-team goals only
    def score(hb: str) -> Tuple[int, int]:
        hg = sum(1 for g in goals_timeline if g["half"] == hb and g["team"] == home_name)
        ag = sum(1 for g in goals_timeline if g["half"] == hb and g["team"] == away_name)
        return hg, ag

    h1, a1 = score("first")
    h2, a2 = score("second")
    ht, at = h1 + h2, a1 + a2

    # player rows
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
            "home_team": home_name,
            "away_team": away_name,
            "tournament": match_payload.get("tournament") or "",
            "round": match_payload.get("round") or "",
            "venue": match_payload.get("venue") or "",
            "kickoffISO8601": match_payload.get("kickoffISO8601") or "",
        },
        "half_event_counts": half_event_counts,  # first/second/unknown event counts
        "score": {
            "first": {"home": h1, "away": a1},
            "second": {"home": h2, "away": a2},
            "total": {"home": ht, "away": at},
        },
        "team_half_counts": team_half_counts,   # per half per team: goals/shots/fouls/subs/timeouts/others
        "goals_timeline": goals_timeline,
        "misc_timeline": misc_timeline,
        "player_stats": rows,
        "charts": [],  # will be filled later (data URIs)
    }

# =========================
# Charts (optional)
# =========================
def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64

def add_charts(stats: Dict[str, Any]) -> None:
    if not (ENABLE_CHARTS and HAVE_MPL):
        return

    m = stats["match"]
    home = m["home_team"]
    away = m["away_team"]

    # 1) team category counts (total)
    def sum_half(h: str, team: str, key: str) -> int:
        return int(stats["team_half_counts"][h][team].get(key, 0))

    keys = ["goals", "shots", "fouls", "subs"]
    labels = ["Goals", "Shots", "Fouls", "Subs"]

    home_vals = [sum_half("first", home, k) + sum_half("second", home, k) for k in keys]
    away_vals = [sum_half("first", away, k) + sum_half("second", away, k) for k in keys]

    fig = plt.figure()
    x = list(range(len(labels)))
    plt.bar([i - 0.2 for i in x], home_vals, width=0.4, label=home)
    plt.bar([i + 0.2 for i in x], away_vals, width=0.4, label=away)
    plt.xticks(x, labels)
    plt.title("チーム比較（主要イベント数）")
    plt.legend()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    stats["charts"].append({"title": "チーム比較（主要イベント数）", "data_uri": uri})

    # 2) top player shots (home/away)
    def player_top(team: str, key: str, topk: int) -> Tuple[List[str], List[int]]:
        rows = [r for r in stats["player_stats"] if r["team"] == team]
        rows.sort(key=lambda r: r.get(key, 0), reverse=True)
        rows = rows[:topk]
        return [r["player"] for r in rows], [int(r.get(key, 0)) for r in rows]

    for team in (home, away):
        names, vals = player_top(team, "shots", CHART_TOPK)
        if not names:
            continue
        fig2 = plt.figure()
        plt.bar(names, vals)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"シュート数 上位{min(CHART_TOPK, len(names))}（{team}）")
        plt.tight_layout()
        uri2 = _fig_to_data_uri(fig2)
        plt.close(fig2)
        stats["charts"].append({"title": f"シュート数（{team}）", "data_uri": uri2})

def charts_markdown(stats: Dict[str, Any]) -> str:
    charts = stats.get("charts") or []
    if not charts:
        return ""
    md = ["\n\n## グラフ（自動生成）\n"]
    for c in charts:
        title = c.get("title", "")
        uri = c.get("data_uri", "")
        if uri:
            md.append(f"### {title}\n\n![]({uri})\n")
    return "\n".join(md)

def stats_tables_markdown(stats: Dict[str, Any]) -> str:
    m = stats["match"]
    home = m["home_team"]
    away = m["away_team"]
    sc = stats["score"]

    lines = []
    lines.append("# 試合分析レポート\n")
    lines.append("## スコア（集計はコード確定）\n")
    lines.append("| | 前半 | 後半 | 合計 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| {home} | {sc['first']['home']} | {sc['second']['home']} | {sc['total']['home']} |")
    lines.append(f"| {away} | {sc['first']['away']} | {sc['second']['away']} | {sc['total']['away']} |")

    # half event counts
    hec = stats["half_event_counts"]
    lines.append("\n## イベント件数（half判定のデバッグ用）\n")
    lines.append("| 前半 | 後半 | half不明 |")
    lines.append("|---:|---:|---:|")
    lines.append(f"| {hec['first']} | {hec['second']} | {hec['unknown']} |")

    # player table (top 12)
    rows = stats.get("player_stats") or []
    if rows:
        top = rows[:12]
        lines.append("\n## 個人指標（上位）\n")
        lines.append("| チーム | 選手 | 得点 | A | シュート | ファウル | IN | OUT |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in top:
            lines.append(
                f"| {r['team']} | {r['player']} | {r['goals']} | {r['assists']} | {r['shots']} | {r['fouls']} | {r['sub_in']} | {r['sub_out']} |"
            )

    return "\n".join(lines) + "\n"

# =========================
# Build messages
# =========================
def build_messages(
    match_payload: Dict[str, Any],
    rag_hits: List[Tuple[str, float, str]],
    stats_json: Dict[str, Any],
    allow_images: bool = True
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
        "テキスト情報（時間・チーム名・選手番号/名前・メモなど）と画像の両方を踏まえて、"
        "systemメッセージの指示に従い、監督・選手向けの技術/戦術レポート（Markdown）を書いてください。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
        "各イベントは「前半/後半/half不明」に分けて提示します。\n"
    )
    user_content.append({"type": "input_text", "text": intro_text})

    # ✅ STATS_JSON を注入（ここが事実の唯一ソース）
    stats_text = json.dumps(stats_json, ensure_ascii=False, indent=2)
    user_content.append({
        "type": "input_text",
        "text": "【STATS_JSON（重要：この中が事実。スコア/得点/集計は改変・再計算禁止）】\n```json\n" + stats_text + "\n```\n"
    })

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
            user_content.append({"type": "input_text", "text": f"\n[1st-{i}] {_build_event_text(ev, home_name, away_name, home_map, away_map)}"})
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
            user_content.append({"type": "input_text", "text": f"\n[2nd-{i}] {_build_event_text(ev, home_name, away_name, home_map, away_map)}"})
            if allow_images:
                sp = ev.get("snapshotPath")
                if sp:
                    url = resolve_snapshot_url(str(sp))
                    if any(x["url"] == url for x in chosen_images):
                        user_content.append({"type": "input_image", "image_url": url})
    else:
        user_content.append({"type": "input_text", "text": "\n【後半イベント】未提供（2ndは0件）。後半について推測で書かないでください。\n"})

    # half不明
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
    # allow "KB001,KB002" or "KB001 KB002"
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

# =========================
# Main API function
# =========================
def generate_match_report_bundle(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    監督/選手向けの Markdown + 表 + (任意)グラフ を返す bundle.
    FastAPI 側で JSON として返したい場合はこちらを使う。
    """
    t0 = time.time()

    # ===== build stats first (facts) =====
    stats = build_stats(match_payload)
    add_charts(stats)

    # ===== RAG query =====
    venue = match_payload.get("venue") or ""
    tournament = match_payload.get("tournament") or ""
    events = match_payload.get("events", []) or []

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = (home.get("name") or "HOME").strip()
    away_name = (away.get("name") or "AWAY").strip()
    home_map = _roster_map(home)
    away_map = _roster_map(away)

    sample_events_text = "\n".join(
        _build_event_text(ev, home_name, away_name, home_map, away_map) for ev in events[:30]
    )
    rag_query = f"{tournament} {venue}\n{sample_events_text}\n戦術 フォーメーション セットプレイ 守備 攻撃"

    rag_hits = rag_search(rag_query, top_k=RAG_TOP_K)
    logger.info("[RAG] top=%d", len(rag_hits))
    for kid, score, ktext in rag_hits:
        logger.info("[RAG] %s score=%.3f text=%s", kid, score, (ktext[:90] + "…") if len(ktext) > 90 else ktext)

    # 1st try: allow images
    messages = build_messages(match_payload, rag_hits, stats_json=stats, allow_images=True)

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
            messages2 = build_messages(match_payload, rag_hits, stats_json=stats, allow_images=False)
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

    # ===== extract markdown report =====
    report_md = extract_report_md(out)
    report_md = strip_rag_used_footer(report_md)  # in case tags leaked into the block

    # ===== prepend tables + append charts =====
    header_md = stats_tables_markdown(stats)
    chart_md = charts_markdown(stats)

    full_md = header_md + "\n" + report_md + "\n" + chart_md

    return {
        "report_md": full_md,
        "rag_used_ids": used_ids,
        "stats": stats,
        "has_charts": bool(stats.get("charts")),
    }

def generate_match_report(match_payload: Dict[str, Any]) -> str:
    """
    互換: 文字列だけ返す（従来の FastAPI の返しが str の場合はこれを使う）
    """
    bundle = generate_match_report_bundle(match_payload)
    return bundle["report_md"]
