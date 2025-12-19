# report_generator.py
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from openai import OpenAI

client = OpenAI()

# =========================
# Logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("report_generator")

# =========================
# Snapshot base url
# =========================
SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", "https://futsal-report-api.onrender.com").rstrip("/")

# =========================
# Config / Prompt
# =========================
def load_prompt() -> str:
    """
    config.json を読み込む。
    Render では CWD がズレることがあるので __file__ 基準にする。
    """
    config_path = Path(__file__).resolve().parent / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        prompt = config.get("system_prompt", "")
        logger.info("[BOOT] Loaded config.json: %s (len=%d)", str(config_path), len(prompt))
        return prompt
    except Exception as e:
        logger.exception("[BOOT] Failed to load config.json: %s", e)
        return ""

SYSTEM_PROMPT_RAW = load_prompt()

# config.json に {futsal_knowledge} が残ってても破綻しないように置換
SYSTEM_PROMPT = (SYSTEM_PROMPT_RAW or "").replace(
    "{futsal_knowledge}",
    "（関連知識はRAGで必要分だけ提示されます）"
)

# “後半が無いなら書くな”を確実に効かせるため、system に追記（保険）
SYSTEM_PROMPT += (
    "\n\n【重要】\n"
    "- 与えられていないイベント（特に後半）について推測・創作しない。\n"
    "- 後半イベントが0件の場合は、後半の描写をせず『後半の記録は未提供』の一文だけにする。\n"
    "- 与えられたイベントに書かれていない得点・失点・時間・選手名は捏造しない。\n"
    "- RAGで提示された知識を使った場合は、該当文末に [KBxxx] を付ける（例: [KB011]）。\n"
)

# =========================
# Knowledge Base (KB)
# =========================
# ※ここはあなたの futsal_knowledge をそのまま流用
KB_TEXTS: List[str] = [
    "フットサルは通常、2×20分のハーフで行われ、インターバルは10分である。",
    "フットサルでは、サッカーと異なり、オフサイドがない。",
    "フットサルのコートはサッカーよりも小さく、サイドラインが短い。",
    "フットサルでは、1チーム5人（ゴールキーパー1人、フィールドプレイヤー4人）で構成される。",
    "フットサルで使われるボールはサッカーよりも小さく、重さが異なる。",
    "フットサルの試合でファウルが累積され、5回を超えると相手チームにフリーキックが与えられる。",

    "4-0フォーメーション（クワトロ）は、守備を重視した戦術で…",
    "3-1フォーメーション（低め）は、3人が守備、1人（ピヴォ）が攻撃起点…",
    "2-2フォーメーション（ボックス型）は、2人守備＋2人攻撃で…",
    "1-3フォーメーション（高め）は、前線で攻撃を強化する…",

    "守備戦術にはゾーンディフェンス、マンツーマンディフェンス、ハイプレスなどがある。",
    "攻撃戦術にはポゼッション重視、カウンターアタック、ウィング攻撃、セットプレイがある。",
    "カウンターアタックは、相手の攻撃から素早く攻撃に転じる戦術で、速いプレイが求められる。",
    "ポゼッションプレイはボール保持を重視し、パス回しで相手を引きつけてスペースを作る。",

    "カットインは、サイドから中央に切れ込んで攻撃を仕掛ける動き。",
    "キックインは、サイドラインを割った際に足で再開するプレイ。",
    "スクリーンは、攻撃側が守備者をブロックしてパスコースを作る。",
    "チョンドンは、キックインからの短いパス→シュートを狙う戦術。",
    "デスマルケは、マークを外してスペースを作る動き。",
    "パワープレーは、GKをフィールドに上げて数的優位で攻撃する。",
    "ハーフは、自陣にコンパクトに構えて守る戦術。",

    "アイソレーション（孤立）は、攻撃側が1対1を作って仕掛ける。",
    "ディフェンシブサードはゴールに近い守備エリア。",
    "ミドルサードは中央の攻守バランスエリア。",
    "アタッキングサードはゴールに近い攻撃エリア。",

    "ピヴォは攻撃の中心で、ボールを収めて起点になる。",
    "アラはサイド担当で攻撃を展開する。",
    "フィクソは守備の中心でライン統率を行う。",
    "ゴールキーパーは守備だけでなく攻撃の起点にもなる。",

    "ゾーンディフェンスはエリアを守る。",
    "マンツーマンディフェンスは人に付く。",
    "ハイプレスは高い位置から圧力をかける。",

    "フリーキックやコーナーのセットプレイでは、ショートやダイレクトを使い分ける。",
    "ゴールキック等の再開から素早く前進するのも重要。",
]

KB_ENTRIES: List[Dict[str, str]] = [
    {"id": f"KB{idx+1:03d}", "text": t} for idx, t in enumerate(KB_TEXTS)
]

KB_EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "text-embedding-3-small")
_KB_EMB: Optional[np.ndarray] = None  # shape=(N, D), float32, normalized


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def init_kb_embeddings() -> None:
    global _KB_EMB
    if _KB_EMB is not None:
        return

    texts = [e["text"] for e in KB_ENTRIES]
    logger.info("[KB] Building embeddings: model=%s, n=%d", KB_EMBED_MODEL, len(texts))

    # まとめて投げる（量が多いなら分割）
    resp = client.embeddings.create(
        model=KB_EMBED_MODEL,
        input=texts,
    )
    emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
    emb = _l2_normalize(emb)
    _KB_EMB = emb

    logger.info("[KB] Embeddings ready: shape=%s", str(_KB_EMB.shape))


def build_rag_query(match_payload: Dict[str, Any], max_chars: int = 1200) -> str:
    """
    試合イベントから埋め込み用クエリを作る（短く濃く）。
    """
    parts: List[str] = []

    # 試合概要
    for k in ["tournament", "round", "venue"]:
        v = match_payload.get(k)
        if v:
            parts.append(str(v))

    events = match_payload.get("events") or []
    if isinstance(events, list):
        for ev in events[:30]:
            t = ev.get("type")
            n = ev.get("note")
            h = ev.get("half")
            if h: parts.append(f"half:{h}")
            if t: parts.append(f"type:{t}")
            if n: parts.append(f"note:{n}")

    q = " / ".join(parts)
    if not q.strip():
        q = "フットサル 試合レポート 戦術 配置 守備 攻撃 セットプレイ"
    return q[:max_chars]


def retrieve_top_k_kb(match_payload: Dict[str, Any], k: int = 4) -> List[Tuple[str, str, float]]:
    """
    OpenAI埋め込み + cosine で Top-k を返す
    戻り: (KBid, text, score)
    """
    init_kb_embeddings()
    assert _KB_EMB is not None

    query = build_rag_query(match_payload)
    q_resp = client.embeddings.create(
        model=KB_EMBED_MODEL,
        input=[query],
    )
    q = np.array([q_resp.data[0].embedding], dtype=np.float32)
    q = _l2_normalize(q)  # shape=(1, D)

    # cosine similarity = dot (normalized)
    scores = (_KB_EMB @ q[0]).astype(np.float32)  # shape=(N,)
    top_idx = np.argsort(-scores)[:k]

    results: List[Tuple[str, str, float]] = []
    for i in top_idx:
        results.append((KB_ENTRIES[int(i)]["id"], KB_ENTRIES[int(i)]["text"], float(scores[int(i)])))
    return results


def extract_used_kb_ids(text: str) -> List[str]:
    """
    出力文中の [KBxxx] を抽出
    """
    ids = re.findall(r"\bKB\d{3}\b", text or "")
    # 重複除去（順序維持）
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# Event formatting
# =========================
def normalize_half(v: Any) -> str:
    """
    受け取りうる half 表記を正規化して "1st"/"2nd"/"unknown" にする
    """
    if v is None:
        return "unknown"

    # 数値系
    if isinstance(v, (int, float)):
        if int(v) == 1:
            return "1st"
        if int(v) == 2:
            return "2nd"
        return "unknown"

    s = str(v).strip().lower()

    first_keys = {
        "1st", "first", "firsthalf", "first_half", "1h", "h1", "前半", "前", "前ハーフ", "前半戦"
    }
    second_keys = {
        "2nd", "second", "secondhalf", "second_half", "2h", "h2", "後半", "後", "後ハーフ", "後半戦"
    }

    s2 = re.sub(r"[\s\-_]", "", s)  # 空白/ハイフン/アンダーバー除去
    if s in first_keys or s2 in first_keys:
        return "1st"
    if s in second_keys or s2 in second_keys:
        return "2nd"

    # "1st half" / "2nd half" 系
    if "1st" in s2 or "first" in s2:
        return "1st"
    if "2nd" in s2 or "second" in s2:
        return "2nd"

    return "unknown"


def get_snapshot_path(ev: Dict[str, Any]) -> Optional[str]:
    """
    snapshotPath のキー揺れを吸収
    """
    return (
        ev.get("snapshotPath")
        or ev.get("snapshot_path")
        or ev.get("snapshotURL")
        or ev.get("snapshotUrl")
        or ev.get("snapshot")
    )


def build_image_url(snapshot_path: str) -> str:
    """
    relative -> absolute
    """
    sp = snapshot_path.strip()
    if sp.startswith("http://") or sp.startswith("https://"):
        return sp
    if not sp.startswith("/"):
        sp = "/" + sp
    return SNAPSHOT_BASE_URL + sp


def _build_event_text(ev: Dict[str, Any]) -> str:
    half_raw = ev.get("half")
    half_norm = normalize_half(half_raw)

    minute_raw = ev.get("minute")
    second_raw = ev.get("second")
    team_side = ev.get("teamSide")  # "home"/"away"/None
    main_no = ev.get("mainPlayerNumber")
    assist_no = ev.get("assistPlayerNumber")
    note = ev.get("note") or ""
    ev_type = ev.get("type") or ""

    # 時間
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

    half_label = {"1st": "前半", "2nd": "後半", "unknown": "ハーフ不明"}[half_norm]
    if minute is not None and second is not None:
        time_str = f"{half_label} {minute:02d}:{second:02d}"
    else:
        time_str = half_label

    # チーム側
    if team_side == "home":
        side_str = "ホーム"
    elif team_side == "away":
        side_str = "アウェイ"
    else:
        side_str = "不明チーム"

    # 選手
    players = []
    if main_no is not None:
        players.append(f"背番号{main_no}")
    if assist_no is not None:
        players.append(f"アシスト背番号{assist_no}")

    players_str = (" / " + "・".join(players)) if players else ""
    note_str = f" / メモ:{note}" if note else ""

    return f"[{time_str}] {side_str} {ev_type}{players_str}{note_str}"


# =========================
# Build multimodal input
# =========================
def build_multimodal_input(match_payload: Dict[str, Any], rag_items: List[Tuple[str, str, float]]) -> List[Dict[str, Any]]:
    venue = match_payload.get("venue") or "会場不明"
    tournament = match_payload.get("tournament") or "大会名不明"
    round_desc = match_payload.get("round") or "ラウンド不明"
    kickoff = match_payload.get("kickoffISO8601") or "日時不明"

    home = match_payload.get("home", {}) or {}
    away = match_payload.get("away", {}) or {}
    home_name = home.get("name", "ホーム")
    away_name = away.get("name", "アウェイ")

    events = match_payload.get("events") or []
    if not isinstance(events, list):
        events = []

    # half別に分ける（unknown も落とさない）
    first_half_events = []
    second_half_events = []
    unknown_half_events = []
    for ev in events:
        hn = normalize_half(ev.get("half"))
        if hn == "1st":
            first_half_events.append(ev)
        elif hn == "2nd":
            second_half_events.append(ev)
        else:
            unknown_half_events.append(ev)

    # snapshot が何件あるか
    snapshot_count = 0
    for ev in events:
        sp = get_snapshot_path(ev)
        if sp and sp != "string" and not str(sp).startswith("data:"):
            snapshot_count += 1

    logger.info(
        "[PAYLOAD] events=%d (1st=%d, 2nd=%d, unknown=%d) snapshots=%d venue=%s",
        len(events), len(first_half_events), len(second_half_events), len(unknown_half_events),
        snapshot_count, venue
    )

    header_text = (
        f"大会: {tournament}\n"
        f"ラウンド: {round_desc}\n"
        f"会場: {venue}\n"
        f"日時(キックオフ想定): {kickoff}\n"
        f"対戦カード: {home_name} vs {away_name}\n"
        f"イベント件数: 前半{len(first_half_events)}件 / 後半{len(second_half_events)}件 / 不明{len(unknown_half_events)}件\n"
    )

    # RAG block（モデルに渡す）＋（ログにも出す）
    rag_lines = []
    for kb_id, text, score in rag_items:
        rag_lines.append(f"{kb_id} (score={score:.3f}): {text}")
    rag_block = "【RAG: 参照用知識】\n" + "\n".join(rag_lines) + "\n"

    # ★ここが「RAGが本当に渡ってる」確認ポイント（Render logs に出る）
    logger.info("[RAG] top=%d", len(rag_items))
    for kb_id, text, score in rag_items:
        logger.info("[RAG] %s score=%.3f text=%s", kb_id, score, text)

    user_content: List[Dict[str, Any]] = []

    intro_text = (
        "以下にフットサルの試合記録（テキスト）と、各イベントに対応する戦術ボード画像を与えます。\n"
        "与えられた情報だけを使ってレポートを書いてください（推測・創作禁止）。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
        "※後半イベントが0件のとき、後半について推測して書かない。\n"
    )
    user_content.append({"type": "input_text", "text": intro_text})

    # 前半
    if first_half_events:
        user_content.append({"type": "input_text", "text": "\n--- 前半 ---\n"})
        for idx, ev in enumerate(first_half_events, start=1):
            user_content.append({"type": "input_text", "text": f"前半イベント{idx}: {_build_event_text(ev)}"})
            sp = get_snapshot_path(ev)
            if sp and sp != "string" and not str(sp).startswith("data:"):
                user_content.append({"type": "input_image", "image_url": build_image_url(str(sp))})

    # 後半：0件なら明示（＝幻覚防止）
    if second_half_events:
        user_content.append({"type": "input_text", "text": "\n--- 後半 ---\n"})
        for idx, ev in enumerate(second_half_events, start=1):
            user_content.append({"type": "input_text", "text": f"後半イベント{idx}: {_build_event_text(ev)}"})
            sp = get_snapshot_path(ev)
            if sp and sp != "string" and not str(sp).startswith("data:"):
                user_content.append({"type": "input_image", "image_url": build_image_url(str(sp))})
    else:
        user_content.append({"type": "input_text", "text": "\n後半の記録は未提供（0件）。※後半について推測して書かない。\n"})

    # half不明：落とさず別枠で渡す（ここが “前半が未提供になる” の最大原因潰し）
    if unknown_half_events:
        user_content.append({"type": "input_text", "text": "\n--- ハーフ不明（入力のhalf表記を要確認） ---\n"})
        for idx, ev in enumerate(unknown_half_events, start=1):
            user_content.append({"type": "input_text", "text": f"不明イベント{idx}: {_build_event_text(ev)}"})
            sp = get_snapshot_path(ev)
            if sp and sp != "string" and not str(sp).startswith("data:"):
                user_content.append({"type": "input_image", "image_url": build_image_url(str(sp))})

    # RAG 知識を user message に “input_text” として追加（重要）
    user_content.append({"type": "input_text", "text": "\n" + rag_block})

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    return messages


# =========================
# Main entry
# =========================
def generate_match_report(match_payload: Dict[str, Any]) -> str:
    """
    FastAPI から呼ばれる想定。
    """
    try:
        rag_items = retrieve_top_k_kb(match_payload, k=4)
        messages = build_multimodal_input(match_payload, rag_items)

        logger.debug("[SEND] SYSTEM_PROMPT(head200)=%s", SYSTEM_PROMPT[:200])
        logger.info("[SEND] messages=%d", len(messages))

        response = client.responses.create(
            model=os.getenv("REPORT_MODEL", "o4-mini"),
            input=messages,
        )

        out = response.output_text or ""
        logger.info("[RESULT] length=%d", len(out))

        # 出力に含まれたKB参照ID（[KBxxx]）をログに出す
        used_ids = extract_used_kb_ids(out)
        if used_ids:
            logger.info("[RAG-USED] ids=%s", ",".join(used_ids))
            # “IDと中身”を Render logs に出したい → ここで展開
            id_to_text = {e["id"]: e["text"] for e in KB_ENTRIES}
            for kb_id in used_ids:
                logger.info("[RAG-USED] %s text=%s", kb_id, id_to_text.get(kb_id, "NOT_FOUND"))
        else:
            logger.info("[RAG-USED] none (モデルが [KBxxx] を付けなかった可能性)")

        return out

    except Exception as e:
        logger.exception("[ERROR] generate_match_report failed: %s", e)
        return "レポート生成中にエラーが発生しました。ログを確認してください。"
