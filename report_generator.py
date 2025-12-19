# report_generator.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from openai import OpenAI


# =========================
# 基本設定
# =========================

client = OpenAI()
SNAPSHOT_BASE_URL = "https://futsal-report-api.onrender.com"

CONFIG_PATH = Path(__file__).with_name("config.json")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 例: text-embedding-3-small / text-embedding-3-large / text-embedding-ada-002
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("futsal_report")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 二重ハンドラ防止
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)

    # uvicorn等の親ロガーに流れないように（好みで）
    logger.propagate = False
    return logger


logger = setup_logger()


# =========================
# config.json（プロンプト）読み込み
# =========================

def load_prompt() -> str:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        prompt = config.get("system_prompt")
        if not prompt:
            raise ValueError("config.json に system_prompt がありません")
        logger.info(f"[PROMPT] Loaded config.json: {CONFIG_PATH}")
        logger.info(f"[PROMPT] prompt_length={len(prompt)} chars")
        return prompt
    except Exception as e:
        logger.exception(f"[PROMPT] Failed to load config.json: {e}")
        # 最低限のフォールバック（必要なら）
        return "あなたはフットサルの試合レポートを書くスポーツライターです。"


SYSTEM_PROMPT_BASE = load_prompt()

# 追加の“強制”ルール（ここはコード側で固定）
SYSTEM_PROMPT_SUFFIX = """
【重要】
- 後半イベントが与えられていない場合、後半について推測・創作しない。必ず「後半の記録は未提供」とだけ書く。
- あなたには「【RAG知識】」として複数の知識断片が与えられる。参考にしたものがあれば、そのIDを末尾1行に必ず出力する。
  形式: [SOURCES] KB001,KB005
  何も使っていないなら: [SOURCES] none
- [SOURCES] 行以外の形式でソースを書かない（本文には出さない）。
""".strip()


def build_system_prompt() -> str:
    # config内に {futsal_knowledge} が残っていても混乱しにくいよう置換（任意）
    prompt = SYSTEM_PROMPT_BASE.replace("{futsal_knowledge}", "（下の【RAG知識】を参照）")
    prompt = prompt.rstrip() + "\n\n" + SYSTEM_PROMPT_SUFFIX + "\n"
    return prompt


# =========================
# 知識ベース（あなたの既存をそのまま）
# =========================

futsal_knowledge: List[str] = [
    # 基本ルール
    "フットサルは通常、2×20分のハーフで行われ、インターバルは10分である。",
    "フットサルでは、サッカーと異なり、オフサイドがない。",
    "フットサルのコートはサッカーよりも小さく、サイドラインが短い。",
    "フットサルでは、1チーム5人（ゴールキーパー1人、フィールドプレイヤー4人）で構成される。",
    "フットサルで使われるボールはサッカーよりも小さく、重さが異なる。",
    "フットサルの試合でファウルが累積され、5回を超えると相手チームにフリーキックが与えられる。",

    # 戦術用語
    "4-0フォーメーション（クワトロ）は、4人が横一列気味に並び、流動的にパス回しして穴を作る攻撃的な形としても使われる（チーム方針で守備重視にもなる）。",
    "3-1フォーメーション（低め）は、後方3枚＋前方1枚（ピヴォ）で、安定した組み立てとピヴォ当てを両立しやすい。",
    "2-2フォーメーション（ボックス型）は、縦関係が作りやすく、守備でもバランスが取りやすい。",
    "1-3フォーメーション（高め）は、前線に厚みを作りやすい一方、背後のリスク管理が重要。",

    "守備戦術にはゾーンディフェンス、マンツーマンディフェンス、ハイプレスなどがある。",
    "攻撃戦術にはポゼッション重視、カウンターアタック、ウィング攻撃、セットプレイがある。",
    "カウンターアタックは、相手の攻撃から素早く攻撃に転じる戦術で、速いプレイが求められる。",
    "ポゼッションプレイはボール保持を重視し、選手同士のパス回しで相手を引きつけてスペースを作り出す。",

    # 特定の戦術的なプレイ
    "カットインは、サイド攻撃選手がサイドラインから中央に切れ込んで攻撃を仕掛ける動き。",
    "キックインは、ボールがサイドラインを越えた場合に、足を使ってボールを再投入するプレイ。",
    "スクリーンは、攻撃側選手がディフェンダーを遮るように体でブロックし、パスやシュートの通り道を作る。",
    "チョンドンは、キックインから短いパス→素早いシュートへつなげるセットプレイの一種。",
    "デスマルケは、相手ディフェンダーのマークを外してフリーになる動き。",
    "パワープレーは、GKをフィールド化して数的優位を作る攻撃。",
    "ハーフは、自陣の半分で守備を固め、ラインをコンパクトに保つ守り方。",

    # エリア用語
    "アイソレーション（孤立）は、攻撃側が1対1を作って仕掛ける状況。",
    "ディフェンシブサードは守備的エリア（自陣深い位置）。",
    "ミドルサードは中央エリア（攻守の切替が多い）。",
    "アタッキングサードは攻撃的エリア（相手ゴールに近い位置）。",

    # ポジション
    "ピヴォは前線でボールを収め、落としや反転でチャンスを作る。",
    "アラはサイドで運ぶ・仕掛ける役割が多い。",
    "フィクソは後方でゲームを整え、カバーリングも担う。",
    "GKは守備だけでなくビルドアップの起点にもなる。",
]


# =========================
# OpenAI Embedding でRAG検索（軽量）
# =========================

KB_IDS: List[str] = [f"KB{i+1:03d}" for i in range(len(futsal_knowledge))]
_KB_EMB: np.ndarray | None = None  # (N, D) normalized
_KB_READY: bool = False


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def embed_texts(texts: List[str]) -> np.ndarray:
    # embeddings APIはまとめて投げられる（KBが少ないので一括でOK）
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return embs


def ensure_kb_embeddings() -> None:
    global _KB_EMB, _KB_READY
    if _KB_READY and _KB_EMB is not None:
        return

    logger.info(f"[RAG] Building KB embeddings... model={EMBED_MODEL} chunks={len(futsal_knowledge)}")
    embs = embed_texts(futsal_knowledge)
    _KB_EMB = _normalize_rows(embs)
    _KB_READY = True
    logger.info(f"[RAG] KB embeddings ready. shape={_KB_EMB.shape}")


def rag_search(query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    ensure_kb_embeddings()
    assert _KB_EMB is not None

    q = embed_texts([query])
    q = _normalize_rows(q)[0]  # (D,)

    # cosine similarity
    sims = _KB_EMB @ q  # (N,)
    k = min(top_k, sims.shape[0])
    idx = np.argsort(-sims)[:k]

    results: List[Dict[str, Any]] = []
    for i in idx:
        results.append({
            "id": KB_IDS[int(i)],
            "score": float(sims[int(i)]),
            "text": futsal_knowledge[int(i)],
        })

    # ★ここが「Render Logsに出したい」部分
    logger.info(f"[RAG] query_len={len(query)} top_k={k}")
    for r in results:
        snippet = r["text"].replace("\n", " ")[:260]
        logger.info(f"[RAG] hit {r['id']} score={r['score']:.4f} text='{snippet}'")

    return results


def build_rag_block(results: List[Dict[str, Any]]) -> str:
    lines = ["【RAG知識】（必要なら参考にしてください）"]
    for r in results:
        # 本文にそのまま渡す（ID付き）
        lines.append(f"[{r['id']}] {r['text']}")
    return "\n".join(lines) + "\n"


# =========================
# 試合イベント整形
# =========================

def _build_event_text(ev: Dict[str, Any]) -> str:
    half = ev.get("half") or ""
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
        time_str = f"{half} {minute:02d}:{second:02d}"
    elif half:
        time_str = half
    else:
        time_str = "時間不明"

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


def build_multimodal_input(match_payload: Dict[str, Any], rag_block_text: str) -> List[Dict[str, Any]]:
    match_id = match_payload.get("matchId")
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
        f"matchId: {match_id}\n"
    )

    events = match_payload.get("events", [])
    first_half_events = [ev for ev in events if ev.get("half") == "1st"]
    second_half_events = [ev for ev in events if ev.get("half") == "2nd"]

    logger.info(f"[DATA] events_total={len(events)} first_half={len(first_half_events)} second_half={len(second_half_events)}")

    user_content: List[Dict[str, Any]] = []

    intro_text = (
        "以下にフットサルの試合記録と、各イベントに対応する戦術ボード画像を与えます。\n"
        "テキスト情報（時間・チーム・選手番号・メモなど）と画像の両方を踏まえて、"
        "systemメッセージの指示に従い、詳細な試合レポートを書いてください。\n\n"
        "【試合概要】\n"
        f"{header_text}\n"
        "【イベント一覧】\n"
        "それぞれのイベントには、可能であれば直後に対応する戦術ボード画像が続きます。\n"
    )

    user_content.append({"type": "input_text", "text": intro_text})

    # ★RAG知識をここで user に追加（＝正しい形）
    user_content.append({"type": "input_text", "text": rag_block_text})

    # --- 前半 ---
    if first_half_events:
        for idx, ev in enumerate(first_half_events, start=1):
            ev_text = _build_event_text(ev)
            ev_header = f"\n--- 前半 イベント {idx} ---\n"
            user_content.append({"type": "input_text", "text": ev_header + ev_text})

            snapshot_path = ev.get("snapshotPath")
            if snapshot_path:
                if snapshot_path == "string" or str(snapshot_path).startswith("data:"):
                    continue
                if str(snapshot_path).startswith("http://") or str(snapshot_path).startswith("https://"):
                    image_url = snapshot_path
                else:
                    snapshot_path = str(snapshot_path)
                    if not snapshot_path.startswith("/"):
                        snapshot_path = "/" + snapshot_path
                    image_url = SNAPSHOT_BASE_URL.rstrip("/") + snapshot_path

                user_content.append({"type": "input_image", "image_url": image_url})

    # --- 後半（なければ明示） ---
    if second_half_events:
        user_content.append({"type": "input_text", "text": "\n--- 後半 ---\n"})
        for idx, ev in enumerate(second_half_events, start=1):
            ev_text = _build_event_text(ev)
            ev_header = f"\n--- 後半 イベント {idx} ---\n"
            user_content.append({"type": "input_text", "text": ev_header + ev_text})

            snapshot_path = ev.get("snapshotPath")
            if snapshot_path:
                if snapshot_path == "string" or str(snapshot_path).startswith("data:"):
                    continue
                if str(snapshot_path).startswith("http://") or str(snapshot_path).startswith("https://"):
                    image_url = snapshot_path
                else:
                    snapshot_path = str(snapshot_path)
                    if not snapshot_path.startswith("/"):
                        snapshot_path = "/" + snapshot_path
                    image_url = SNAPSHOT_BASE_URL.rstrip("/") + snapshot_path

                user_content.append({"type": "input_image", "image_url": image_url})
    else:
        # ★これが「後半創作」をかなり止める
        user_content.append({"type": "input_text", "text": "\n【後半イベント】未提供（記録なし）\n"})

    messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": build_system_prompt()}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


# =========================
# 出力の [SOURCES] をログに取り、本文から削除
# =========================

_SOURCES_PREFIX = "[SOURCES]"

def split_sources_line(text: str) -> Tuple[str, List[str]]:
    lines = text.splitlines()
    used_ids: List[str] = []
    kept: List[str] = []

    for line in lines:
        s = line.strip()
        if s.startswith(_SOURCES_PREFIX):
            used_ids = re.findall(r"KB\d{3}", s)
        else:
            kept.append(line)

    cleaned = "\n".join(kept).strip()
    return cleaned, used_ids


# =========================
# メイン：レポート生成（FastAPIから呼ばれる）
# =========================

def build_rag_query(match_payload: Dict[str, Any]) -> str:
    """試合内容から検索クエリを自動生成（適当に長すぎないように）"""
    home = (match_payload.get("home") or {}).get("name", "")
    away = (match_payload.get("away") or {}).get("name", "")
    tournament = match_payload.get("tournament", "") or ""
    venue = match_payload.get("venue", "") or ""

    events = match_payload.get("events", []) or []
    parts = [tournament, venue, f"{home} vs {away}"]

    # イベントのタイプ・メモを少し混ぜる（最大20件くらい）
    for ev in events[:20]:
        t = ev.get("type") or ""
        note = ev.get("note") or ""
        if t:
            parts.append(str(t))
        if note:
            parts.append(str(note))

    q = " / ".join([p for p in parts if p])
    return q[:2000]


def generate_match_report(match_payload: Dict[str, Any]) -> str:
    # 1) RAG検索 → ログ出力（ここで ID/中身が Render logs に出る）
    query = build_rag_query(match_payload)
    rag_hits = rag_search(query, top_k=RAG_TOP_K)
    rag_block = build_rag_block(rag_hits)

    # 2) LLM入力を構築
    messages = build_multimodal_input(match_payload, rag_block)

    # 3) LLM呼び出し
    logger.info("[LLM] calling responses.create model=o4-mini")
    response = client.responses.create(
        model="o4-mini",
        input=messages,
    )
    out = response.output_text or ""
    logger.info(f"[LLM] output_len={len(out)} chars")

    # 4) [SOURCES] を抽出してログに出す（本文からは消す）
    cleaned, used_ids = split_sources_line(out)

    if used_ids:
        # 使ったと言っているIDをログへ
        logger.info(f"[RAG] model_cited={','.join(used_ids)}")
        # そのIDの本文（ヒットの中から一致を探す）
        hit_map = {h["id"]: h for h in rag_hits}
        for kid in used_ids:
            if kid in hit_map:
                snippet = hit_map[kid]["text"].replace("\n", " ")[:260]
                logger.info(f"[RAG] cited_text {kid}: '{snippet}'")
            else:
                logger.info(f"[RAG] cited_text {kid}: (not in top_k hits)")
    else:
        logger.info("[RAG] model_cited=none")

    return cleaned
