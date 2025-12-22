import csv
import io
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

from .models import Analysis


def analyses_to_csv(analyses: Iterable[Analysis]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["id", "created_at", "label_pred", "prob_ai", "text", "model_name", "segments"]
    )
    for a in analyses:
        writer.writerow(
            [
                a.id,
                a.created_at.isoformat() if a.created_at else "",
                a.label_pred,
                f"{a.prob_ai:.4f}",
                a.text,
                a.model_name,
                a.segments_json,
            ]
        )
    return output.getvalue()


def analyses_to_pdf(analyses: Iterable[Analysis]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = [Paragraph("AI Text Detector Report", styles["Heading1"])]

    table_data = [["ID", "Created", "Label", "Prob AI", "Preview"]]
    for a in analyses:
        preview = a.text[:80] + ("..." if len(a.text) > 80 else "")
        table_data.append(
            [
                str(a.id),
                a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "",
                str(a.label_pred),
                f"{a.prob_ai:.3f}",
                preview,
            ]
        )

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
