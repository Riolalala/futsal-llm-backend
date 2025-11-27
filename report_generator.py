# report_generator.py
# -*- coding: utf-8 -*-
"""
o4-mini を使ってフットサルの試合レポートを生成するモジュール。

- Swift 側の LLMPayload（matchId, home/away, events など）に対応する dict を受け取り、
  それをもとに日本語の試合レポートを生成する。
- FastAPI 側 (main.py) から generate_match_report() を呼び出して使う前提。
"""

import json
from typing import Dict, Any
from openai import OpenAI

# OPENAI_API_KEY は環境変数から読み込まれる想定
client = OpenAI()

# ===== プロンプト =====

SYSTEM_PROMPT = """\
あなたはフットサルの試合レポートを書くスポーツライターです。
以下の条件で日本語の試合レポートを書いてください。

- 試合の概要（大会名やカテゴリがあればそれも）を最初に一文でまとめる
- スコアとゴールの時間・得点者を時系列で整理する
- 前半・後半それぞれでどのような流れだったかを描写する
- ハイライトシーンを2〜3個ピックアップして、状況や流れの変化が分かるように書く
- 最後に「この試合の収穫と今後の課題」を短くまとめる
- だいたい 600〜1000文字程度
"""


def build_user_prompt(match_payload: Dict[str, Any]) -> str:
    """
    Swift の LLMPayload と同じ構造の dict を受け取り、
    User メッセージ用のテキストに変換する。
    """
    # JSONを整形（人間が読んでもわかるようにしておく）
    pretty = json.dumps(match_payload, ensure_ascii=False, indent=2)

    prompt = (
        "以下はフットサルの試合記録です。\n"
        "この情報をもとに、systemメッセージの条件に従って日本語の試合レポートを書いてください。\n\n"
        "試合記録(JSON):\n"
        f"{pretty}\n"
    )
    return prompt


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
    user_prompt = build_user_prompt(match_payload)

    # Responses API で o4-mini を呼び出す
    response = client.responses.create(
        model="o4-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    # OpenAI Python SDK のショートカット: すべてのテキスト出力をまとめたプロパティ
    return response.output_text
