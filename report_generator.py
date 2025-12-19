# report_generator.py
# -*- coding: utf-8 -*-
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# Logging (Renderで確実に出す)
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # ★ uvicorn等が先にlogger設定していても上書き
)
logger = logging.getLogger("report")

client = OpenAI()

SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", "https://futsal-report-api.onrender.com")

# =========================
# config.json を確実に読む
# =========================
def load_prompt() -> str:
    """
    - Renderだとカレントがズレることがあるので、
      `__file__` 基準で config.json を読むのが安全。
    - どのファイルを読んだかをログに必ず出す。
    """
    env_path = os.getenv("CONFIG_PATH")  # 任意で差し替え可能
    config_path = Path(env_path) if env_path else Path(__file__).with_name("config.json")

    logger.debug(f"[config] cwd={os.getcwd()}")
    logger.debug(f"[config] config_path={config_path.resolve()}")
    logger.debug(f"[config] exists={config_path.exists()}")

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path.resolve()}")

    raw = config_path.read_text(encoding="utf-8")
    config = json.loads(raw)

    if "system_prompt" not in config or not isinstance(config["system_prompt"], str):
        raise ValueError("config.json must contain string field: system_prompt")

    # 更新確認用：mtime と先頭だけ表示
    stat = config_path.stat()
    logger.debug(f"[config] mtime={stat.st_mtime}")
    prompt = config["system_prompt"]
    logger.debug("[config] system_prompt(head 200)=\n" + prompt[:200])

    return prompt


SYSTEM_PROMPT = load_prompt()

# =========================
# フットサル知識（RAG用）
# =========================
futsal_knowledge = [
    "フットサルは通常、2×20分のハーフで行われ、インターバルは10分である。",
    "フットサルでは、サッカーと異なり、オフサイドがない。",
    "フットサルのコートはサッカーよりも小さく、サイドラインが短い。",
    "フットサルでは、1チーム5人（ゴールキーパー1人、フィールドプレイヤー4人）で構成される。",
    "フットサルで使われるボールはサッカーよりも小さく、重さが異なる。",
    "フットサルの試合でファウルが累積され、5回を超えると相手チームにフリーキックが与えられる。",
    "4-0フォーメーション（クワトロ）は、…（省略）",
    "3-1フォーメーション（低め）は、…（省略）",
    "2-2フォーメーション（ボックス型）は、…（省略）",
    "1-3フォーメーション（高め）は、…（省略）",
    "守備戦術にはゾーンディフェンス、マンツーマンディフェンス、ハイプレスなどがある。",
    "攻撃戦術にはポゼッション重視、カウンターアタック、ウィング攻撃、セットプレイがある。",
    "カットインは、サイド攻撃選手がサイドラインから中央に切れ込んで攻撃を仕掛ける動き。",
    "キックインは、ボールがサイドラインを越えた場合に、足を使ってボールを再投入するプレイ。",
    "スクリーンは、攻撃側選手がディフェンダーを遮るように体で守備選手をブロックし、パスの通り道を作る。",
    "チョンドン（キックインからの短いパスでシュートを試みる）は、速攻を意識したキックインからの戦術。",
    "デスマルケは、相手ディフェンダーからマークを外し、自由に動くことでスペースを作る戦術。",
    "パワープレーは、ゴールキーパーをフィールドプレイヤーとして使用し、攻撃を強化する戦術。",
    "ハーフは、自陣の半分で守備を固め、相手の攻撃を防ぐ戦術。特に守備ラインをコンパクトに保つ。",
    "アイソレーション（孤立）は、攻撃側が相手のディフェンダーを1対1で対決させる戦術。",
    "ディフェンシブサードは、相手の攻撃を防ぐための守備的エリアで、ゴールに近い位置。",
    "ミドルサードは、攻守のバランスが求められる中央エリア。",
    "アタッキングサードは、ゴールに近い攻撃的エリアで、得点を狙う場所。",
]

# embedding / faiss (起動時に作る)
model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_embeddings = model.encode(futsal_knowledge)
index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
index.add(np.array(knowledge_embeddings))


# =========================
# half 正規化（"1st"/"前半" など混在対策）
# =========================
def normalize_half(h: Any) -> Optional[str]:
    if h is None:
        return None
    s = str(h).strip().lower()
    if s in ["1st", "first", "h1", "前半", "firsthalf", "1"]:
        return "1st"
    if s in ["2nd", "second", "h2", "後半", "secondhalf", "2"]:
        return "2nd"
    return None

def half_label_jp(norm: Optional[str]) -> str:
    if norm == "1st":
        return "前半"
    if norm == "2nd":
        return "後半"
    return "時間不明"


def _build_event_text(ev: Dict[str, Any]) -> str:
    half_norm = normalize_half(ev.get("half"))
    half_jp = half_label_jp(half_norm)

    minute_raw = ev.get("minute")
    second_raw = ev.get("second")
    team_side = ev.get("teamSide")  # home/away
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
        time_str = f"{half_jp} {minute:02d}:{second:02d}"
    else:
        time_str = half_jp

    if team_side == "home":
        side_str = "ホーム"
    elif team_side == "away":
        side_str = "アウェイ"
    else:
        side_str = "チーム不明"

    players_str = ""
    if main_no is not None:
        players_str += f" 主な関与: #{main_no}"
    if assist_no is not None:
        players_str += f" / アシスト: #{assist_no}"

    note_str = f" メモ: {note}" if note else ""
    return f"[{time_str}] {side_str} {ev_type}。{players_str}{note_str}".strip()


def build_multimodal_input(match_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    venue = match_payload.get("venue") or "会場不明"
    tournament = match_payload.get("tournament") or "大会名不明"
    round_desc = match_payload.get("round") or "ラウンド不明"
    kickoff = match_payload.get("kickoffISO8601") or "日時不明"

    home = match_payload.get("home", {})
    away = match_payload.get("away", {})
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
    for ev in events:
        ev["_half_norm"] = normalize_half(ev.get("half"))

    first_half_events = [ev for ev in events if ev.get("_half_norm") == "1st"]
    second_half_events = [ev for ev in events if ev.get("_half_norm") == "2nd"]

    user_content: List[Dict[str, Any]] = []

    intro_text = (
        "以下にフットサルの試合記録と、各イベントに対応する戦術ボード画像を与えます。\n"
        "【重要】与えられたイベント以外を推測・創作しないでください。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
    )
    user_content.append({"type": "input_text", "text": intro_text})

    # ---- 前半 ----
    user_content.append({"type": "input_text", "text": "\n=== 前半イベント ===\n"})
    if first_half_events:
        for idx, ev in enumerate(first_half_events, start=1):
            ev_text = _build_event_text(ev)
            user_content.append({"type": "input_text", "text": f"\n--- 前半 イベント {idx} ---\n{ev_text}"})

            snapshot_path = ev.get("snapshotPath")
            if snapshot_path:
                if snapshot_path == "string" or str(snapshot_path).startswith("data:"):
                    continue
                sp = str(snapshot_path)
                if sp.startswith("http://") or sp.startswith("https://"):
                    image_url = sp
                else:
                    if not sp.startswith("/"):
                        sp = "/" + sp
                    image_url = SNAPSHOT_BASE_URL.rstrip("/") + sp
                user_content.append({"type": "input_image", "image_url": image_url})
    else:
        user_content.append({"type": "input_text", "text": "前半イベント: 未提供（0件）\n"})

    # ---- 後半 ----（★未提供を明示）
    user_content.append({"type": "input_text", "text": "\n=== 後半イベント ===\n"})
    if second_half_events:
        for idx, ev in enumerate(second_half_events, start=1):
            ev_text = _build_event_text(ev)
            user_content.append({"type": "input_text", "text": f"\n--- 後半 イベント {idx} ---\n{ev_text}"})

            snapshot_path = ev.get("snapshotPath")
            if snapshot_path:
                if snapshot_path == "string" or str(snapshot_path).startswith("data:"):
                    continue
                sp = str(snapshot_path)
                if sp.startswith("http://") or sp.startswith("https://"):
                    image_url = sp
                else:
                    if not sp.startswith("/"):
                        sp = "/" + sp
                    image_url = SNAPSHOT_BASE_URL.rstrip("/") + sp
                user_content.append({"type": "input_image", "image_url": image_url})
    else:
        # ★モデルに「後半は書くな」を二重に伝える
        user_content.append({"type": "input_text", "text": "後半イベント: 未提供（0件）\n※後半について推測して書かない。\n"})

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    return messages


def retrieve_rag_text(match_payload: Dict[str, Any], k: int = 4) -> str:
    """
    試合内容からクエリを作って、近い知識を k 件取り出し、文字列に整形する。
    """
    events = match_payload.get("events", []) or []
    # クエリは「試合概要 + 先頭イベント数個」で十分
    sample_texts = []
    for ev in events[:10]:
        sample_texts.append(_build_event_text(ev))

    query = " ".join(sample_texts) if sample_texts else "フットサル 試合レポート 戦術 まとめ"
    qv = model.encode([query])
    D, I = index.search(np.array(qv), k=k)

    hits = [futsal_knowledge[i] for i in I[0]]
    logger.debug(f"[RAG] query(head 120)={query[:120]}")
    logger.debug(f"[RAG] hit_indices={I[0].tolist()}")
    logger.debug(f"[RAG] hit_texts={hits}")

    rag_text = "【参照用フットサル知識（RAG）】\n" + "\n".join([f"- {t}" for t in hits])
    return rag_text


def generate_match_report(match_payload: Dict[str, Any]) -> str:
    messages = build_multimodal_input(match_payload)

    # ---- RAGを「userのinput_text」として追加（★これが正しい形）----
    rag_text = retrieve_rag_text(match_payload, k=4)
    messages.append({
        "role": "user",
        "content": [{"type": "input_text", "text": rag_text}],
    })

    # ---- 送信内容の確認ログ（でかすぎるので先頭だけ）----
    logger.debug("[send] SYSTEM_PROMPT(head 200)=\n" + SYSTEM_PROMPT[:200])
    logger.debug("[send] messages_count=%d", len(messages))
    logger.debug("[send] last_user_message(head 300)=\n" + rag_text[:300])

    response = client.responses.create(
        model="o4-mini",
        input=messages,
    )

    out = response.output_text or ""
    logger.debug("[result] length=%d", len(out))
    logger.debug("[result] head 400=\n" + out[:400])

    return out
