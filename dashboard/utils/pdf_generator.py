"""
PDF report generator for Lloyds SME Intelligence Platform.
Uses fpdf2 — pure Python, no system dependencies.
"""
from __future__ import annotations
import textwrap
from datetime import datetime

from fpdf import FPDF


def _sanitize(text: str) -> str:
    """Strip markdown and replace characters outside Latin-1 so Helvetica won't error."""
    replacements = {
        "–": "-", "—": "-",   # en-dash, em-dash
        "‘": "'", "’": "'",   # left/right single quotes
        "“": '"', "”": '"',   # left/right double quotes
        "•": "*", "…": "...", # bullet, ellipsis
        " ": " ",                  # non-breaking space
    }
    for ch, repl in replacements.items():
        text = text.replace(ch, repl)
    # Strip markdown
    text = (text.replace("**", "").replace("##", "")
                .replace("#", "").replace("---", ""))
    # Drop anything still outside Latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")

LLOYDS_GREEN  = (0, 106, 77)
LLOYDS_LIGHT  = (230, 245, 238)
TEXT_DARK     = (30, 30, 30)
TEXT_GREY     = (100, 100, 100)
WHITE         = (255, 255, 255)


class _SMEReport(FPDF):
    def header(self):
        self.set_fill_color(*LLOYDS_GREEN)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*WHITE)
        self.set_xy(10, 4)
        self.cell(0, 10, "Lloyds SME Intelligence Platform  |  Group 33", ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_xy(0, 6)
        self.cell(200, 6, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}", align="R")
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*TEXT_GREY)
        self.cell(0, 6,
                  "CONFIDENTIAL — For internal Lloyds Banking Group use only. "
                  "ML models trained on Companies House public data.",
                  align="C")
        self.set_text_color(*TEXT_DARK)

    def section_title(self, title: str):
        self.ln(4)
        self.set_fill_color(*LLOYDS_GREEN)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"  {title}", ln=True, fill=True)
        self.set_text_color(*TEXT_DARK)
        self.ln(2)

    def kv_row(self, key: str, value: str, shade: bool = False):
        if shade:
            self.set_fill_color(*LLOYDS_LIGHT)
        else:
            self.set_fill_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(65, 7, f"  {key}", fill=True)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 7, value, ln=True, fill=True)

    def score_row(self, label: str, prob: float, shade: bool = False):
        if prob >= 0.7:
            status, color = "HIGH", (0, 150, 80)
        elif prob >= 0.4:
            status, color = "MEDIUM", (200, 130, 0)
        else:
            status, color = "LOW", (200, 40, 40)

        if shade:
            self.set_fill_color(*LLOYDS_LIGHT)
        else:
            self.set_fill_color(*WHITE)
        self.set_font("Helvetica", "", 9)
        self.cell(80, 8, f"  {label}", fill=True)
        self.cell(40, 8, f"{prob:.1%}", fill=True)
        self.set_text_color(*color)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 8, status, ln=True, fill=True)
        self.set_text_color(*TEXT_DARK)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXT_DARK)
        # Strip markdown and replace characters outside Latin-1 range
        clean = _sanitize(text)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                self.ln(2)
                continue
            # Wrap long lines to page width
            wrapped = textwrap.wrap(line, width=100)
            for wl in wrapped:
                self.multi_cell(0, 5, wl)
        self.ln(2)

    def shap_features(self, label: str, features: list[tuple[str, float]]):
        if not features:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*LLOYDS_GREEN)
        self.cell(0, 5, f"  {label.capitalize()}", ln=True)
        self.set_text_color(*TEXT_DARK)
        for i, (fname, fval) in enumerate(features[:6]):
            arrow = "+" if fval >= 0 else "-"
            shade = i % 2 == 0
            if shade:
                self.set_fill_color(*LLOYDS_LIGHT)
            else:
                self.set_fill_color(*WHITE)
            self.set_font("Helvetica", "", 8)
            self.cell(110, 5, f"    {i+1}. {fname}", fill=True)
            color = (0, 130, 60) if fval >= 0 else (180, 30, 30)
            self.set_text_color(*color)
            self.set_font("Helvetica", "B", 8)
            self.cell(0, 5, f"{arrow}{abs(fval):.4f}", ln=True, fill=True)
            self.set_text_color(*TEXT_DARK)
        self.ln(2)


def build_pdf(
    sector: str,
    growth: float,
    risk: float,
    lending: float,
    report_text: str,
    top_features: dict[str, list[tuple[str, float]]] | None = None,
    company_name: str = "Unknown Company",
) -> bytes:
    """
    Build a PDF report and return the raw bytes.

    Parameters
    ----------
    sector       : BCB sector string
    growth/risk/lending : probabilities 0-1
    report_text  : AI-generated narrative (markdown accepted, stripped for PDF)
    top_features : {label: [(feature_name, shap_value), ...]}
    company_name : optional label shown on the cover
    """
    pdf = _SMEReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_text_color(*TEXT_DARK)

    # ── Cover title ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*LLOYDS_GREEN)
    pdf.ln(4)
    pdf.cell(0, 12, "SME Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*TEXT_GREY)
    pdf.cell(0, 7, company_name, ln=True, align="C")
    pdf.ln(6)

    # ── Company overview ───────────────────────────────────────────────────
    pdf.section_title("COMPANY OVERVIEW")
    pdf.kv_row("Sector",           sector,                         shade=True)
    pdf.kv_row("Report date",      datetime.now().strftime("%d %B %Y"), shade=False)
    pdf.kv_row("ML model",         "XGBoost (sample-trained, 100k companies)", shade=True)
    pdf.kv_row("Training data",    "3,145,434 UK Companies House records",    shade=False)

    # ── Prediction scores ──────────────────────────────────────────────────
    pdf.section_title("PREDICTION SCORES")
    pdf.set_fill_color(*LLOYDS_GREEN)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(80, 7, "  Signal", fill=True)
    pdf.cell(40, 7, "Probability", fill=True)
    pdf.cell(0,  7, "Status", fill=True, ln=True)
    pdf.set_text_color(*TEXT_DARK)

    pdf.score_row("Growth Opportunity",  growth,  shade=True)
    pdf.score_row("Risk Signal",         risk,    shade=False)
    pdf.score_row("Lending Need Proxy",  lending, shade=True)

    # ── SHAP features ──────────────────────────────────────────────────────
    if top_features:
        pdf.section_title("KEY DRIVERS (SHAP FEATURE IMPORTANCE)")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*TEXT_GREY)
        pdf.cell(0, 5, "  Positive values push prediction higher; negative values push it lower.", ln=True)
        pdf.set_text_color(*TEXT_DARK)
        pdf.ln(2)
        for lbl in ["growth", "risk", "lending"]:
            feats = top_features.get(lbl, [])
            if feats:
                pdf.shap_features(lbl, feats)

    # ── AI narrative ───────────────────────────────────────────────────────
    pdf.section_title("AI ANALYSIS (GENERATED BY LLM GATEWAY)")
    pdf.body_text(report_text)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    pdf.section_title("DISCLAIMER")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*TEXT_GREY)
    pdf.multi_cell(0, 5,
        "This report is generated by machine learning models and a generative AI system. "
        "Predictions are probabilistic and should be used as decision-support tools only, "
        "not as the sole basis for lending or investment decisions. "
        "Data sourced from UK Companies House public bulk download. "
        "Lloyds Banking Group Group 33 Dissertation Project.")
    pdf.set_text_color(*TEXT_DARK)

    return bytes(pdf.output())


def pdf_download_button(
    report_text: str,
    sector: str,
    growth: float,
    risk: float,
    lending: float,
    top_features: dict | None = None,
    company_name: str = "SME",
    key: str = "pdf_download",
):
    """Render a Streamlit download button that delivers the PDF."""
    import streamlit as st

    try:
        pdf_bytes = build_pdf(
            sector=sector,
            growth=growth,
            risk=risk,
            lending=lending,
            report_text=report_text,
            top_features=top_features,
            company_name=company_name,
        )
        filename = f"lloyds_sme_report_{sector.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=key,
        )
    except Exception as e:
        st.error(f"PDF generation error: {e}")
