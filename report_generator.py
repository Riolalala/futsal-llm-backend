# report_generator.py
# -*- coding: utf-8 -*-
"""
o4-mini を使ってフットサルの試合レポートを生成するモジュール（テキスト＋画像版）。

- Swift 側の LLMPayload（matchId, home/away, events など）に対応する dict を受け取り、
  それをもとに日本語の試合レポートを生成する。
- events[].snapshotPath には、FastAPI の /upload_snapshot から返された
  「/snapshots/<matchId>/<eventId>.png」のような相対パスが入る想定。
"""

import json
import os
from typing import Dict, Any, List
from openai import OpenAI

client = OpenAI()

# 画像URLのベース（Render 本番のURL）。環境変数で上書きも可
SNAPSHOT_BASE_URL = os.getenv(
    "SNAPSHOT_BASE_URL",
    "https://futsal-report-api.onrender.com",
)

# ===== プロンプト =====

SYSTEM_PROMPT = """\
あなたはフットサルの試合レポートを書くスポーツライターです。
与えられた試合記録（テキスト情報）と戦術ボード画像をもとに、
日本語で読みやすい試合レポートを書いてください。

- 試合の概要（大会名・カテゴリ・対戦カード・会場など）を最初に一文でまとめる
- スコアとゴールの時間・得点者を時系列で整理する
- 前半・後半それぞれでどのような流れだったかを描写する
- 画像から分かる「配置」「マークの付き方」「数的優位/劣位」「狙い」なども言及する
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


def build_multimodal_input(match_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Responses API に渡す input（system + user）を構築する。
    - system: SYSTEM_PROMPT（input_text）
    - user: 試合概要テキスト + 各イベントのテキスト & 画像（snapshotPath があれば input_image）
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

    # --- user.content を組み立てる（テキスト＋画像） ---
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

    user_content.append({
        "type": "input_text",
        "text": intro_text,
    })

    for idx, ev in enumerate(events, start=1):
        # まずテキスト
        ev_text = _build_event_text(ev)
        ev_header = f"\n--- イベント {idx} ---\n"
        user_content.append({
            "type": "input_text",
            "text": ev_header + ev_text,
        })

        # 次に画像（あれば）
        snapshot_path = ev.get("snapshotPath")

        if not snapshot_path:
            continue

        # 🔴 ここでフィルタリング：変な値("string" など)は無視
        if snapshot_path == "string" or snapshot_path == "string.":
            continue

        # フルURLの組み立て
        if snapshot_path.startswith("http://") or snapshot_path.startswith("https://"):
            image_url = snapshot_path
        elif snapshot_path.startswith("/"):
            # 例: "/snapshots/<matchId>/<eventId>.png"
            image_url = SNAPSHOT_BASE_URL.rstrip("/") + snapshot_path
        else:
            # "snapshots/..." のようにスラッシュなしで来た場合もケア
            image_url = SNAPSHOT_BASE_URL.rstrip("/") + "/" + snapshot_path

        user_content.append({
            "type": "input_image",
            "image_url": image_url,
        })

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def generate_match_report(match_payload: Dict[str, Any]) -> str:
    """
    FastAPI から呼び出される想定の関数。
    """
    messages = build_multimodal_input(match_payload)

    response = client.responses.create(
        model="o4-mini",
        input=messages,
    )

    return response.output_text
