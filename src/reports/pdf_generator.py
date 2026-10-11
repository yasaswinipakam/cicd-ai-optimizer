"""Single-page PDF build-analysis report generation."""

from __future__ import annotations

from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _value(value) -> str:
    return "-" if value is None or value == "" else str(value)


def _section_heading(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["Heading2"])


def _explanation_rows(explanation) -> list[list[str]]:
    """Normalize the existing JSON-safe local-SHAP response for a PDF table."""
    if not isinstance(explanation, (list, tuple)):
        return []

    rows: list[list[str]] = []
    for item in explanation:
        if not isinstance(item, dict) or not item.get("feature"):
            continue
        value = _value(item.get("value"))
        contribution = _value(item.get("shap_value"))
        direction = _value(item.get("direction")).replace("_", " ")
        rows.append([_value(item["feature"]), value, contribution, direction])
    return rows


def generate_build_report_pdf(build_data: dict, output_path: str) -> None:
    """Create a polished one-page PDF combining all build-analysis outputs."""
    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=38,
        bottomMargin=38,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#0f172a")))
    styles.add(ParagraphStyle(name="SmallFooter", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER))
    styles["Heading2"].textColor = colors.HexColor("#075985")
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 5

    prediction = build_data.get("prediction", {})
    anomaly = build_data.get("anomaly", {})
    root_cause = build_data.get("root_cause", {})
    suggestion = build_data.get("optimization_suggestion")
    explanation_rows = _explanation_rows(build_data.get("explanation"))
    failed = prediction.get("predicted_status") == "failed" or str(build_data.get("build_status", "")).lower() == "failed"
    status_color = "#b91c1c" if prediction.get("predicted_status") == "failed" else "#15803d"

    story = [
        Paragraph("CI/CD Build Analysis Report", styles["ReportTitle"]),
        Spacer(1, 8),
        Table(
            [
                ["Pipeline", _value(build_data.get("pipeline_id"))],
                ["Build ID", _value(build_data.get("build_id"))],
                ["Build timestamp", _value(build_data.get("timestamp"))],
                ["Generated at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
            ],
            colWidths=[1.35 * inch, 5.5 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0c4a6e")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        _section_heading("Prediction", styles),
        Paragraph(
            f"Predicted status: <font color='{status_color}'><b>{_value(prediction.get('predicted_status')).upper()}</b></font><br/>"
            f"Failure probability: <b>{float(prediction.get('failure_probability', 0)) * 100:.1f}%</b><br/>"
            f"Confidence: <b>{float(prediction.get('confidence', 0)) * 100:.1f}%</b>",
            styles["BodyText"],
        ),
        _section_heading("Anomaly Detection", styles),
        Paragraph(
            f"Unusual pre-build input profile: <b>{'Yes' if anomaly.get('is_anomaly') else 'No'}</b><br/>"
            f"Anomaly score: <b>{_value(anomaly.get('anomaly_score'))}</b><br/>"
            "This is not a prediction that the build will fail.",
            styles["BodyText"],
        ),
    ]
    if failed:
        terms = build_data.get("top_terms") or root_cause.get("top_terms") or []
        root_cause_text = _value(root_cause.get("likely_cause")) if root_cause.get("likely_cause") else "No failed-test log available for root-cause analysis."
        terms_text = _value(", ".join(terms)) if terms else "No failed-test log available for root-cause analysis."
        story += [
            _section_heading("Root Cause Analysis", styles),
            Paragraph(
                f"Cluster label: <b>{root_cause_text}</b><br/>"
                f"Top contributing terms: <b>{terms_text}</b>",
                styles["BodyText"],
            ),
        ]
    if suggestion:
        story += [
            _section_heading("Optimization Suggestion", styles),
            Table(
                [[Paragraph(
                    f"<b>{_value(suggestion.get('category'))}: {_value(suggestion.get('title'))}</b><br/>"
                    f"{_value(suggestion.get('suggestion'))}", styles["BodyText"]
                )]],
                colWidths=[6.85 * inch],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#7dd3fc")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]),
            ),
        ]
    if explanation_rows:
        story += [
            _section_heading("Prediction Explanation", styles),
            Paragraph(
                "Top local feature contributions for this prediction. Positive contributions increase failure risk; negative contributions decrease it.",
                styles["BodyText"],
            ),
            Spacer(1, 4),
            Table(
                [["Feature", "Value", "Contribution", "Impact"], *explanation_rows],
                colWidths=[1.8 * inch, 1.15 * inch, 1.35 * inch, 2.55 * inch],
                repeatRows=1,
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0c4a6e")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 5),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]),
            ),
        ]
    story += [Spacer(1, 14), Paragraph("Generated by AI-Driven CI/CD Pipeline Optimization System - BMSCE Capstone Project", styles["SmallFooter"])]
    document.build(story)
