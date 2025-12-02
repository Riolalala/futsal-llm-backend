# report_generator.py
# -*- coding: utf-8 -*-
"""
o4-mini を使ってフットサルの試合レポートを生成するモジュール（テキストのみ版）。

- Swift 側の LLMPayload（matchId, home/away, events など）に対応する dict を受け取り、
  それをもとに日本語の試合レポートを生成する。
- まずは安定動作を優先して、画像（snapshotPath）は一切使わず、
  試合記録のテキスト情報だけでレポートを生成する。
"""

import json
from typing import Dict, Any, List
from openai import OpenAI

client = OpenAI()

# ===== プロンプト =====

SYSTEM_PROMPT = """\
あなたはフットサルの試合レポートを書くスポーツライターです。
与えられた試合記録（テキスト情報）をもとに、
日本語で読みやすい試合レポートを書いてください。

- 試合の概要（大会名・カテゴリ・対戦カード・会場など）を最初に一文でまとめる
- スコアとゴールの時間・得点者を時系列で整理する
- 前半・後半それぞれでどのような流れだったかを描写する
- ハイライトシーンを2〜3個ピックアップして、状況や流れの変化が分かるように書く
- 最後に「この試合の収穫と今後の課題」を短くまとめる
- だいたい 600〜1000文字程度
"""


def _build_event_text(ev: Dict[str, Any]) -> str:
    """
    1イベントぶんを人間向けテキストに整形する。
    （半分・時間・チーム・選手番号・メモなど）
    """
    half = ev.get("half") or ""
    minute_raw = ev.get("minute")
    second_raw = ev.get("second")
    team_side = ev.get("teamSide")  # "home" or "away" or None
    main_no = ev.get("mainPlayerNumber")
    assist_no = ev.get("assistPlayerNumber")
    note = ev.get("note") or ""
    ev_type = ev.get("type") or ""

    # --- 時間表記を安全に作る ---
    minute = None
    second = None
    try:
        if minute_raw is not None:
            minute = int(minute_raw)
        if second_raw is not None:
            second = int(second_raw)
    except (TypeError, ValueError):
        # 変な値が来たらあきらめて「時間不明」に落とす
        minute = None
        second = None

    if minute is not None and second is not None:
        time_str = f"{half} {minute:02d}:{second:02d}"
    elif half:
        time_str = half
    else:
        time_str = "時間不明"

    # --- チーム側 ---
    if team_side == "home":
        side_str = "ホームチーム"
    elif team_side == "away":
        side_str = "アウェイチーム"
    else:
        side_str = "チーム不明"

    # --- 選手情報 ---
    players_str = ""
    if main_no is not None:
        players_str += f" 主な関与選手: 背番号{main_no}"
    if assist_no is not None:
        if players_str:
            players_str += f"、アシスト: 背番号{assist_no}"
        else:
            players_str += f" アシスト: 背番号{assist_no}"

    note_str = f" メモ: {note}" if note else ""

    return f"[{time_str}] {side_str} の {ev_type}。{players_str}{note_str}".strip()



def build_text_only_input(match_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Responses API に渡す input（system + user）を構築する。
    - system: SYSTEM_PROMPT（input_text）
    - user: 試合概要テキスト + 各イベントのテキスト（画像は使わない）
    """
    # --- 試合の概要 ---
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

    events = match_payload.get("events", [])

    # --- user.content を組み立てる（テキストのみ） ---
    lines: List[str] = []

    lines.append("以下はフットサルの試合記録です。")
    lines.append("この情報をもとに、systemメッセージの指示に従い、詳細な試合レポートを書いてください。")
    lines.append("")
    lines.append("【試合概要】")
    lines.append(header_text.strip())
    lines.append("")
    lines.append("【イベント一覧】")

    for idx, ev in enumerate(events, start=1):
        ev_text = _build_event_text(ev)
        lines.append(f"\n--- イベント {idx} ---")
        lines.append(ev_text)

    full_text = "\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": full_text},
            ],
        },
    ]
    return messages


def generate_match_report(match_payload: Dict[str, Any]) -> str:
    """
    FastAPI から呼び出される想定の関数。

    Parameters
    ----------
    match_payload : dict
        Swift 側で組み立てられた LLMPayload を Pydantic の model_dump()
        などで dict にしたもの。

    Returns
    -------
    str
        o4-mini が生成した試合レポート（日本語テキスト）。
    """
    messages = build_text_only_input(match_payload)

    response = client.responses.create(
        model="o4-mini",  # ← ここを有効なモデル名に
        input=messages,
    )

    return response.output_text
