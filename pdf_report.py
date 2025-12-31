# pdf_report.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def build_pdf(out_pdf: str, stats: dict, report_md_text: str, chart_paths: list[str]):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    story = []

    m = stats["match"]
    title = f'{m["tournament"]} {m["round"]}  {m["home_team"]} vs {m["away_team"]}'
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f'会場: {m.get("venue","")}  kickoff: {m.get("kickoffISO8601","")}', styles["Normal"]))
    story.append(Spacer(1, 12))

    # スコア表
    sc = stats["score"]
    data = [
        ["", "前半", "後半", "合計"],
        [m["home_team"], sc["first"]["home"], sc["second"]["home"], sc["total"]["home"]],
        [m["away_team"], sc["first"]["away"], sc["second"]["away"], sc["total"]["away"]],
    ]
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # グラフ
    for p in chart_paths:
        story.append(Image(p, width=480, height=240))
        story.append(Spacer(1, 12))

    # ※ Markdown をそのまま綺麗にPDF化するには追加処理が必要
    # まずは report_md_text を Paragraph にざっくり流す（簡易）
    story.append(Paragraph(report_md_text.replace("\n", "<br/>"), styles["Normal"]))

    doc.build(story)