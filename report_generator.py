# report_generator.py
# -*- coding: utf-8 -*-

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from openai import OpenAI

# =========================
# Settings (env)
# =========================
SNAPSHOT_BASE_URL = os.getenv("SNAPSHOT_BASE_URL", "https://futsal-report-api.onrender.com")
CHAT_MODEL = os.getenv("CHAT_MODEL", "o4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

client = OpenAI()

# =========================
# Logging (Render logs に確実に出す)
# =========================
logger = logging.getLogger("report_generator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# =========================
# Knowledge Base (ID付き)
# =========================
_KB_TEXTS: List[str] = [
    # 基本ルール
    "フットサルは通常、2×20分のハーフで行われ、インターバルは10分である。",
    "フットサルでは、サッカーと異なり、オフサイドがない。",
    "フットサルのコートはサッカーよりも小さく、サイドラインが短い。",
    "フットサルでは、1チーム5人（ゴールキーパー1人、フィールドプレイヤー4人）で構成される。",
    "フットサルで使われるボールはサッカーよりも小さく、重さが異なる。",
    "フットサルの試合でファウルが累積され、5回を超えると相手チームにフリーキックが与えられる。",

    # 戦術用語（※文章は長くてもOK）
    "4-0フォーメーション（クワトロ）は、4人がフラットに広がりパス回しで崩す攻撃的配置。相手の守備を横に揺さぶり、アイソレーションを作りやすい。",
    "3-1フォーメーション（低め）は、3人が後方で安定し、1人（ピヴォ）を起点に前線で収める。守備は安定しやすく、攻撃はピヴォ当てから展開する。",
    "2-2フォーメーション（ボックス型）は、2人守備＋2人攻撃の近い距離で連動しやすい。縦関係が作りやすく、中央を使った崩しやカウンターにも移りやすい。",
    "1-3フォーメーション（高め）は、前線でプレッシャーをかけやすく攻撃枚数を確保しやすいが、背後のリスク管理が重要。",

    "守備戦術にはゾーンディフェンス、マンツーマンディフェンス、ハイプレスなどがある。",
    "攻撃戦術にはポゼッション重視、カウンターアタック、ウィング攻撃、セットプレイがある。",
    "カウンターアタックは、相手の攻撃から素早く攻撃に転じる戦術で、速いプレイが求められる。",
    "ポゼッションプレイはボール保持を重視し、選手同士のパス回しで相手を引きつけてスペースを作り出す。",

    # 特定プレー
    "カットインは、サイド攻撃選手がサイドラインから中央に切れ込んで攻撃を仕掛ける動き。",
    "キックインは、ボールがサイドラインを越えた場合に、足を使ってボールを再投入するプレイ。",
    "スクリーンは、攻撃側選手がディフェンダーを遮るように体でブロックし、味方のフリーを作る。",
    "チョンドン（キックインからの短いパスでシュートを試みる）は、速攻を意識したキックイン戦術。",
    "デスマルケは、マークを外して受け直し、パスコースやシュートコースを作る動き。",
    "パワープレーは、GKをフィールドプレイヤー化して数的優位を作る戦術。",
    "ハーフは、自陣の半分で守備を固め、コンパクトにラインを保つ戦術。",

    # エリア用語
    "アイソレーション（孤立）は、1対1を作って個で崩す状況づくり。",
    "ディフェンシブサードはゴールに近い守備的エリア。",
    "ミドルサードは中央の攻守バランスが重要なエリア。",
    "アタッキングサードはゴールに近い攻撃的エリア。",

    # ポジション
    "ピヴォは攻撃の中心となる選手で、前線で収めて起点になる。",
    "アラはサイドを担当して攻撃を展開する。",
    "フィクソは守備の中心で、ビルドアップにも関与する。",
    "ゴールキーパーは守備だけでなく、ビルドアップの起点としても重要。",

    # セットプレイ
    "フリーキックやコーナーのセットプレイでは、ショートやダイレクトを使い分けて守備のズレを作る。",
    "ゴールキック等では安全な前進（保持）と、一気の前進（裏）を状況で選ぶ。",
]

KB: List[Dict[str, str]] = []
for i, t in enumerate(_KB_TEXTS, start=1):
    KB.append({"id": f"KB{i:03d}", "text": t})

_KB_VECS: Optional[np.ndarray] = None  # (N, d) normalized
_KB_READY: bool = False


# =========================
# Config (system prompt)
# =========================
_DEFAULT_SYSTEM_PROMPT = """\
あなたはフットサルの試合レポートを書くスポーツライターです。
与えられた試合記録（テキスト情報）と戦術ボード画像をもとに、日本語で読みやすい試合レポートを書いてください。

- 試合の概要（大会名・カテゴリ・対戦カード・会場など）を最初に一文でまとめる
- スコアとゴールの時間・得点者を時系列で整理する
- シュートやファウルなども整理する
- 画像がある場合のみ、画像から分かる「配置」「マークの付き方」「数的優位/劣位」「狙い」なども言及する
- ハイライトシーンを2〜3個ピックアップして、状況や流れの変化が分かるように書く
- 最後に「この試合の収穫と今後の課題」を短くまとめる
- だいたい 600文字程度

重要:
- 後半イベントが与えられていない場合、後半について推測・創作して書かない（後半は触れない）
"""


def _resolve_config_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    p = Path(CONFIG_PATH)
    if not p.is_absolute():
        p = base_dir / p
    return p


def load_system_prompt() -> str:
    p = _resolve_config_path()
    logger.info("[CONFIG] cwd=%s", str(Path.cwd()))
    logger.info("[CONFIG] config_path=%s exists=%s", str(p), p.exists())
    if not p.exists():
        logger.warning("[CONFIG] config.json not found -> use default prompt")
        return _DEFAULT_SYSTEM_PROMPT

    try:
        config = json.loads(p.read_text(encoding="utf-8"))
        s = config.get("system_prompt")
        if not isinstance(s, str) or not s.strip():
            logger.warning("[CONFIG] system_prompt missing/empty -> use default prompt")
            return _DEFAULT_SYSTEM_PROMPT
        return s
    except Exception as e:
        logger.exception("[CONFIG] failed to load config.json: %s", e)
        return _DEFAULT_SYSTEM_PROMPT


def _augment_system_prompt(base: str) -> str:
    """
    ユーザーのpromptに「内部タグ」要求を追加（出力からは除去して返す）
    """
    extra = """
【内部ルール（ユーザーには表示しない）】
- 与えたRAG知識（KBxxx）を使った場合、レポート本文にIDは書かず、
  末尾に次の1行を必ず付ける：
  <RAG_USED>KB001,KB002</RAG_USED>
  使わなかった場合は：
  <RAG_USED>none</RAG_USED>
"""
    # {futsal_knowledge} が残っていても壊れないように置換だけする
    base = base.replace("{futsal_knowledge}", "（関連知識はRAGで追記します）")
    return base.strip() + "\n\n" + extra.strip()


SYSTEM_PROMPT = _augment_system_prompt(load_system_prompt())
logger.info("[PROMPT] loaded length=%d", len(SYSTEM_PROMPT))


# =========================
# Helpers: half normalize / snapshot extract
# =========================
def normalize_half(v: Any) -> Optional[str]:
    """
    入力の揺れを吸収して '1st' / '2nd' / None にする
    """
    if v is None:
        return None

    # int / float / bool
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if int(v) == 1:
            return "1st"
        if int(v) == 2:
            return "2nd"
        return None

    s = str(v).strip()
    if not s:
        return None

    s_low = s.lower()

    # 1st patterns
    if s_low in {"1st", "1", "1h", "1sthalf", "first", "firsthalf"}:
        return "1st"
    if "前半" in s or s_low in {"zenhan", "1half"}:
        return "1st"

    # 2nd patterns
    if s_low in {"2nd", "2", "2h", "2ndhalf", "second", "secondhalf"}:
        return "2nd"
    if "後半" in s or s_low in {"kouhan", "2half"}:
        return "2nd"

    # more fuzzy
    if re.search(r"\b1(st)?\b", s_low) and "half" in s_low:
        return "1st"
    if re.search(r"\b2(nd)?\b", s_low) and "half" in s_low:
        return "2nd"

    return None


def extract_snapshot_path(ev: Dict[str, Any]) -> Optional[str]:
    """
    Swift側のキー揺れを吸収して snapshotPath を取り出す
    """
    keys = [
        "snapshotPath",
        "snapshot_path",
        "snapshotURL",
        "snapshotUrl",
        "boardSnapshotPath",
        "boardImagePath",
        "imagePath",
        "image_url",
        "imageUrl",
    ]
    for k in keys:
        v = ev.get(k)
        if isinstance(v, str) and v.strip():
            if v == "string" or v.startswith("data:"):
                return None
            return v.strip()
    return None


def to_image_url(snapshot_path: str) -> str:
    """
    相対パスなら SNAPSHOT_BASE_URL を付与
    """
    if snapshot_path.startswith("http://") or snapshot_path.startswith("https://"):
        return snapshot_path
    sp = snapshot_path
    if not sp.startswith("/"):
        sp = "/" + sp
    return SNAPSHOT_BASE_URL.rstrip("/") + sp


# =========================
# Embeddings RAG
# =========================
def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    OpenAI Embeddings -> (len(texts), d)
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs


def _ensure_kb_ready() -> None:
    global _KB_VECS, _KB_READY
    if _KB_READY and _KB_VECS is not None:
        return

    try:
        texts = [x["text"] for x in KB]
        vecs = _embed_texts(texts)
        # normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        _KB_VECS = vecs
        _KB_READY = True
        logger.info("[RAG] KB embeddings ready: n=%d dim=%d model=%s", vecs.shape[0], vecs.shape[1], EMBEDDING_MODEL)
    except Exception as e:
        logger.exception("[RAG] failed to build KB embeddings: %s", e)
        _KB_VECS = None
        _KB_READY = False


def retrieve_kb(query: str, top_k: int) -> List[Tuple[str, float, str]]:
    """
    return list of (KBid, score, text)
    cosine similarity (because normalized)
    """
    _ensure_kb_ready()
    if _KB_VECS is None:
        return []

    q_vec = _embed_texts([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    scores = _KB_VECS @ q_vec  # (N,)
    idx = np.argsort(-scores)[: max(1, top_k)]
    out: List[Tuple[str, float, str]] = []
    for i in idx:
        kb = KB[int(i)]
        out.append((kb["id"], float(scores[int(i)]), kb["text"]))
    return out


# =========================
# Prompt build
# =========================
def _build_event_text(ev: Dict[str, Any]) -> str:
    half_raw = ev.get("half")
    half = normalize_half(half_raw) or "unknown"

    minute_raw = ev.get("minute")
    second_raw = ev.get("second")

    team_side = ev.get("teamSide")  # "home" / "away" / None
    main_no = ev.get("mainPlayerNumber")
    assist_no = ev.get("assistPlayerNumber")
    note = ev.get("note") or ""
    ev_type = ev.get("type") or ""

    # safe time
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
    else:
        time_str = half

    # side
    if team_side == "home":
        side_str = "ホーム"
    elif team_side == "away":
        side_str = "アウェイ"
    else:
        side_str = "不明"

    players = []
    if main_no is not None:
        players.append(f"主:#{main_no}")
    if assist_no is not None:
        players.append(f"助:#{assist_no}")

    players_str = (" " + " ".join(players)) if players else ""
    note_str = f" メモ:{note}" if note else ""

    return f"[{time_str}]({side_str}) {ev_type}{players_str}{note_str}".strip()


def build_multimodal_messages(match_payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    returns: (messages, meta)
    meta includes counts for logging
    """
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

    first: List[Dict[str, Any]] = []
    second: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []

    # snapshot counting (we count paths)
    snapshot_paths: List[str] = []

    unknown_half_samples: List[str] = []

    for ev in events:
        h_norm = normalize_half(ev.get("half"))
        if h_norm == "1st":
            first.append(ev)
        elif h_norm == "2nd":
            second.append(ev)
        else:
            unknown.append(ev)
            # sample raw values for log
            raw = ev.get("half")
            if raw is not None and len(unknown_half_samples) < 8:
                unknown_half_samples.append(str(raw))

        sp = extract_snapshot_path(ev)
        if sp:
            snapshot_paths.append(sp)

    meta = {
        "matchId": str(match_id) if match_id is not None else None,
        "venue": venue,
        "tournament": tournament,
        "home": home_name,
        "away": away_name,
        "events_total": len(events),
        "events_1st": len(first),
        "events_2nd": len(second),
        "events_unknown": len(unknown),
        "unknown_half_samples": unknown_half_samples,
        "snapshots": len(snapshot_paths),
    }

    # ----- RAG query text (短く要約して埋め込み) -----
    # 事件が多いと長くなるので、先頭だけ使う（情報は十分）
    ev_texts_for_query = []
    for ev in events[:40]:
        ev_type = str(ev.get("type") or "")
        note = str(ev.get("note") or "")
        half = str(ev.get("half") or "")
        ev_texts_for_query.append(f"{half}:{ev_type}:{note}".strip(":"))
    query = f"{tournament} {venue} {home_name} vs {away_name} " + " ".join(ev_texts_for_query)
    query = query[:2500]  # safety

    rag_hits = retrieve_kb(query, RAG_TOP_K)
    meta["rag_top_k"] = len(rag_hits)

    # ----- user content -----
    header_text = (
        f"大会: {tournament}\n"
        f"ラウンド: {round_desc}\n"
        f"会場: {venue}\n"
        f"日時(キックオフ想定): {kickoff}\n"
        f"対戦カード: {home_name} vs {away_name}\n"
        f"イベント数: {len(events)}（前半={len(first)} / 後半={len(second)} / half不明={len(unknown)}）\n"
        f"戦術ボード画像: {len(snapshot_paths)}枚\n"
    )

    # 後半が0なら明示（モデルの“補完”を止める）
    half_notice = ""
    if len(second) == 0:
        half_notice = "【重要】後半イベントは 0件（未提供）です。後半について推測・創作して書かないでください。\n"

    rag_text = ""
    if rag_hits:
        rag_lines = [f"- {kb_id}: {txt}" for (kb_id, _score, txt) in rag_hits]
        rag_text = (
            "【RAG: 関連知識（参考）】\n"
            "※本文にKBのIDは書かないでください。必要なら内容だけ自然に使ってください。\n"
            + "\n".join(rag_lines)
            + "\n"
        )
    else:
        rag_text = "【RAG: 関連知識】該当なし\n"

    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "input_text", "text": "【試合概要】\n" + header_text + "\n" + half_notice + "\n" + rag_text})

    # --- 前半 ---
    if first:
        user_content.append({"type": "input_text", "text": "\n--- 前半イベント ---\n"})
        for idx, ev in enumerate(first, start=1):
            user_content.append({"type": "input_text", "text": f"前半#{idx} " + _build_event_text(ev)})
            sp = extract_snapshot_path(ev)
            if sp:
                user_content.append({"type": "input_image", "image_url": to_image_url(sp)})

    # --- 後半（ある場合のみ） ---
    if second:
        user_content.append({"type": "input_text", "text": "\n--- 後半イベント ---\n"})
        for idx, ev in enumerate(second, start=1):
            user_content.append({"type": "input_text", "text": f"後半#{idx} " + _build_event_text(ev)})
            sp = extract_snapshot_path(ev)
            if sp:
                user_content.append({"type": "input_image", "image_url": to_image_url(sp)})

    # --- half不明（ログ的には重要なので“別枠”で渡す：正規化できないときの保険） ---
    # ここを渡す/渡さないは好みだが、今は「情報落ち」を防ぐため渡す（ただし明確にunknown扱いにする）
    if unknown:
        user_content.append({"type": "input_text", "text": "\n--- half不明イベント（表記揺れの可能性） ---\n"})
        for idx, ev in enumerate(unknown, start=1):
            user_content.append({"type": "input_text", "text": f"unknown#{idx} " + _build_event_text(ev)})
            sp = extract_snapshot_path(ev)
            if sp:
                user_content.append({"type": "input_image", "image_url": to_image_url(sp)})

    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    # ----- logs (RAG hits) -----
    logger.info(
        "[PAYLOAD] events=%d (1st=%d, 2nd=%d, unknown=%d) snapshots=%d venue=%s",
        meta["events_total"], meta["events_1st"], meta["events_2nd"], meta["events_unknown"], meta["snapshots"], venue
    )
    if unknown_half_samples:
        logger.info("[PAYLOAD] unknown half samples=%s", ",".join(unknown_half_samples))

    if rag_hits:
        logger.info("[RAG] top=%d", len(rag_hits))
        for kb_id, score, txt in rag_hits:
            logger.info("[RAG] %s score=%.3f text=%s", kb_id, score, (txt[:120] + "…") if len(txt) > 120 else txt)
    else:
        logger.info("[RAG] top=0 (no hits or KB not ready)")

    logger.info("[SEND] messages=%d (system+user)", len(messages))
    return messages, meta


def _extract_rag_used_ids(text: str) -> List[str]:
    """
    <RAG_USED>KB001,KB002</RAG_USED> を抽出
    """
    m = re.search(r"<RAG_USED>(.*?)</RAG_USED>", text, flags=re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.lower() == "none" or raw == "":
        return []
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    # KBxxx形式だけに絞る
    ids = [x for x in ids if re.fullmatch(r"KB\d{3}", x)]
    return ids


def _strip_internal_tags(text: str) -> str:
    """
    ユーザーに返す本文から内部タグを除去
    """
    text = re.sub(r"\n?<RAG_USED>.*?</RAG_USED>\n?", "\n", text, flags=re.DOTALL)
    return text.strip()


# =========================
# Public: generate report
# =========================
def generate_match_report(match_payload: Dict[str, Any]) -> str:
    messages, _meta = build_multimodal_messages(match_payload)

    response = client.responses.create(
        model=CHAT_MODEL,
        input=messages,
    )

    out = response.output_text or ""
    logger.info("[RESULT] length=%d", len(out))

    used_ids = _extract_rag_used_ids(out)
    if used_ids:
        logger.info("[RAG-USED] ids=%s", ",".join(used_ids))
        # 使ったIDの本文もログに出す
        used_map = {kb["id"]: kb["text"] for kb in KB}
        for kb_id in used_ids:
            txt = used_map.get(kb_id, "")
            logger.info("[RAG-USED] %s text=%s", kb_id, (txt[:140] + "…") if len(txt) > 140 else txt)
    else:
        logger.info("[RAG-USED] ids=none")

    return _strip_internal_tags(out)
