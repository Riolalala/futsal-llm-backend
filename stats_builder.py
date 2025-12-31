# stats_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

def _half_norm(h: Optional[str]) -> str:
    if not h:
        return "unknown"
    s = str(h).lower()
    if s in ("1st","first","前半","1"):
        return "first"
    if s in ("2nd","second","後半","2"):
        return "second"
    return "unknown"

def _sec(minute: Optional[int], second: Optional[int]) -> Optional[int]:
    if minute is None or second is None:
        return None
    return int(minute) * 60 + int(second)

def build_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    home_name = payload["home"]["name"]
    away_name = payload["away"]["name"]

    # 背番号→名前
    home_map = {p["number"]: p.get("name","") for p in payload["home"].get("players", [])}
    away_map = {p["number"]: p.get("name","") for p in payload["away"].get("players", [])}

    events = payload.get("events", [])

    # カウント
    half_event_counts = {"first":0, "second":0, "unknown":0}
    team_stats = {
        home_name: defaultdict(int),
        away_name: defaultdict(int),
        "チーム不明": defaultdict(int),
    }
    half_stats = {
        "first": {home_name: defaultdict(int), away_name: defaultdict(int), "チーム不明": defaultdict(int)},
        "second": {home_name: defaultdict(int), away_name: defaultdict(int), "チーム不明": defaultdict(int)},
        "unknown": {home_name: defaultdict(int), away_name: defaultdict(int), "チーム不明": defaultdict(int)},
    }

    # 個人
    player_stats = defaultdict(lambda: defaultdict(int))  # key: "TEAM|#NO Name"

    goals_timeline = []   # 得点経過（チームが判明するゴールのみ）
    misc_timeline  = []   # シュート/ファウル/交代等

    def team_of(e) -> str:
        ts = e.get("teamSide")
        if ts == "home":
            return home_name
        if ts == "away":
            return away_name
        return "チーム不明"

    def player_label(team: str, no: Optional[int]) -> Optional[str]:
        if no is None:
            return None
        if team == home_name:
            name = home_map.get(no, "")
        elif team == away_name:
            name = away_map.get(no, "")
        else:
            name = ""
        return f"{team}|#{no} {name}".strip()

    def is_goal(t: str) -> bool:
        s = t.lower()
        return ("goal" in s) or ("ゴール" in t) or ("得点" in t)

    def is_shot(t: str) -> bool:
        s = t.lower()
        return ("shot" in s) or ("シュート" in t)

    def is_foul(t: str) -> bool:
        s = t.lower()
        return ("foul" in s) or ("ファウル" in t)

    def is_sub(t: str) -> bool:
        s = t.lower()
        return ("substitution" in s) or ("交代" in t)

    def is_timeout(t: str) -> bool:
        s = t.lower()
        return ("timeout" in s) or ("タイムアウト" in t)

    for e in events:
        half = _half_norm(e.get("half"))
        half_event_counts[half] += 1

        tname = team_of(e)
        etype = e.get("type") or e.get("action") or ""
        etype_str = str(etype)

        # チーム集計
        team_stats[tname][etype_str] += 1
        half_stats[half][tname][etype_str] += 1

        mm, ss = e.get("minute"), e.get("second")
        when_sec = _sec(mm, ss)

        main_no = e.get("mainPlayerNumber")
        ast_no  = e.get("assistPlayerNumber")

        if is_goal(etype_str):
            # ゴールはスコア計算対象（チーム不明は除外）
            if tname != "チーム不明":
                goals_timeline.append({
                    "half": half,
                    "minute": mm, "second": ss,
                    "time_sec": when_sec,
                    "team": tname,
                    "scorer_no": main_no,
                    "scorer_name": (home_map.get(main_no,"") if tname==home_name else away_map.get(main_no,"")) if main_no is not None else "",
                    "assist_no": ast_no,
                    "assist_name": (home_map.get(ast_no,"") if tname==home_name else away_map.get(ast_no,"")) if ast_no is not None else "",
                    "note": e.get("note"),
                    "snapshotPath": e.get("snapshotPath"),
                })

                # 個人集計
                pl = player_label(tname, main_no)
                if pl: player_stats[pl]["goals"] += 1
                al = player_label(tname, ast_no)
                if al: player_stats[al]["assists"] += 1
            else:
                misc_timeline.append({
                    "half": half, "minute": mm, "second": ss,
                    "time_sec": when_sec,
                    "team": tname,
                    "type": etype_str,
                    "note": "チーム不明ゴール（スコア計算外）",
                })
        else:
            # それ以外のイベントは雑にタイムラインへ
            if is_shot(etype_str):
                pl = player_label(tname, main_no)
                if pl: player_stats[pl]["shots"] += 1
            if is_foul(etype_str):
                pl = player_label(tname, main_no)
                if pl: player_stats[pl]["fouls"] += 1
            if is_sub(etype_str):
                # OUT=main / IN=assist の想定
                outl = player_label(tname, main_no)
                inl  = player_label(tname, ast_no)
                if outl: player_stats[outl]["sub_out"] += 1
                if inl:  player_stats[inl]["sub_in"] += 1
            if is_timeout(etype_str):
                team_stats[tname]["timeouts"] += 1

            misc_timeline.append({
                "half": half, "minute": mm, "second": ss,
                "time_sec": when_sec,
                "team": tname,
                "type": etype_str,
                "main_no": main_no,
                "assist_no": ast_no,
                "note": e.get("note"),
                "snapshotPath": e.get("snapshotPath"),
            })

    # スコア（ゴールタイムラインから）
    def score_by_half(h: str) -> Tuple[int,int]:
        hg = sum(1 for g in goals_timeline if g["half"]==h and g["team"]==home_name)
        ag = sum(1 for g in goals_timeline if g["half"]==h and g["team"]==away_name)
        return hg, ag

    h1,a1 = score_by_half("first")
    h2,a2 = score_by_half("second")
    ht,at = (h1+h2), (a1+a2)

    # player_stats を表にしやすい配列へ
    player_rows = []
    for key, d in player_stats.items():
        team, rest = key.split("|", 1)
        player_rows.append({
            "team": team,
            "player": rest,
            "goals": int(d.get("goals",0)),
            "assists": int(d.get("assists",0)),
            "shots": int(d.get("shots",0)),
            "fouls": int(d.get("fouls",0)),
            "sub_in": int(d.get("sub_in",0)),
            "sub_out": int(d.get("sub_out",0)),
        })

    # 時系列ソート
    goals_timeline.sort(key=lambda x: (x["half"], x["time_sec"] if x["time_sec"] is not None else 10**9))
    misc_timeline.sort(key=lambda x: (x["half"], x["time_sec"] if x["time_sec"] is not None else 10**9))

    return {
        "match": {
            "home_team": home_name,
            "away_team": away_name,
            "tournament": payload.get("tournament"),
            "round": payload.get("round"),
            "venue": payload.get("venue"),
            "kickoffISO8601": payload.get("kickoffISO8601"),
        },
        "half_event_counts": half_event_counts,
        "score": {
            "first":  {"home": h1, "away": a1},
            "second": {"home": h2, "away": a2},
            "total":  {"home": ht, "away": at},
        },
        "goals_timeline": goals_timeline,
        "misc_timeline": misc_timeline,
        "player_stats": player_rows,
        # team_stats/half_stats は辞書のままだと大きいので必要なら整形して渡す
    }
