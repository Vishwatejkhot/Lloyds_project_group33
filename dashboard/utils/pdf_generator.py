"""
PDF report generator for Lloyds SME Intelligence Platform.
Uses fpdf2 — pure Python, no system dependencies.
"""
from __future__ import annotations
import textwrap
from datetime import datetime

from fpdf import FPDF

LLOYDS_GREEN = (0, 106, 77)
LLOYDS_LIGHT = (230, 245, 238)
TEXT_DARK    = (30, 30, 30)
TEXT_GREY    = (100, 100, 100)
WHITE        = (255, 255, 255)

L_MARGIN  = 12
R_MARGIN  = 12
PAGE_W    = 210       # A4
EFF_W     = PAGE_W - L_MARGIN - R_MARGIN   # 186mm usable width


def _sanitize(text: str) -> str:
    """Replace characters outside Latin-1 so Helvetica won't error."""
    replacements = {
        "–": "-",  "—": "-",   # en-dash, em-dash
        "‘": "'",  "’": "'",   # left/right single quotes
        "“": '"',  "”": '"',   # left/right double quotes
        "•": "*",  "…": "...", # bullet, ellipsis
        " ": " ",                   # non-breaking space
    }
    for ch, repl in replacements.items():
        text = text.replace(ch, repl)
    text = (text.replace("**", "").replace("##", "")
                .replace("# ", "").replace("---", ""))
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _SMEReport(FPDF):
    def header(self):
        # Green banner
        self.set_fill_color(*LLOYDS_GREEN)
        self.rect(0, 0, PAGE_W, 16, "F")
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*WHITE)
        self.set_xy(L_MARGIN, 3)
        self.cell(EFF_W * 0.65, 10, "Lloyds SME Intelligence Platform  |  Group 33")
        self.set_font("Helvetica", "", 8)
        self.set_xy(L_MARGIN + EFF_W * 0.65, 5)
        self.cell(EFF_W * 0.35, 6,
                  datetime.now().strftime("%d %b %Y  %H:%M"),
                  align="R")
        self.set_xy(L_MARGIN, 16)
        self.set_text_color(*TEXT_DARK)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*TEXT_GREY)
        self.set_x(L_MARGIN)
        self.cell(EFF_W, 6,
                  "CONFIDENTIAL - For internal Lloyds Banking Group use only. "
                  "ML models trained on Companies House public data.",
                  align="C")
        self.set_text_color(*TEXT_DARK)

    # ── Layout helpers ──────────────────────────────────────────────────────
    def section_bar(self, title: str):
        self.ln(4)
        self.set_fill_color(*LLOYDS_GREEN)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.set_x(L_MARGIN)
        self.cell(EFF_W, 7, f"  {title}", fill=True, ln=True)
        self.set_text_color(*TEXT_DARK)
        self.ln(1)

    def two_col(self, label: str, value: str, shade: bool = False):
        fill_c = LLOYDS_LIGHT if shade else WHITE
        self.set_fill_color(*fill_c)
        self.set_x(L_MARGIN)
        col1 = EFF_W * 0.35
        col2 = EFF_W * 0.65
        self.set_font("Helvetica", "B", 9)
        self.cell(col1, 7, f"  {label}", fill=True)
        self.set_font("Helvetica", "", 9)
        self.cell(col2, 7, _sanitize(value), fill=True, ln=True)

    def score_row(self, label: str, prob: float, shade: bool = False):
        if prob >= 0.7:   status, sc = "HIGH",   (0, 150, 80)
        elif prob >= 0.4: status, sc = "MEDIUM",  (200, 130, 0)
        else:             status, sc = "LOW",     (200, 40, 40)

        fill_c = LLOYDS_LIGHT if shade else WHITE
        self.set_fill_color(*fill_c)
        self.set_x(L_MARGIN)
        col1, col2, col3 = EFF_W * 0.50, EFF_W * 0.25, EFF_W * 0.25
        self.set_font("Helvetica", "", 9)
        self.cell(col1, 8, f"  {label}", fill=True)
        self.cell(col2, 8, f"{prob:.1%}", fill=True)
        self.set_text_color(*sc)
        self.set_font("Helvetica", "B", 9)
        self.cell(col3, 8, status, fill=True, ln=True)
        self.set_text_color(*TEXT_DARK)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXT_DARK)
        clean = _sanitize(text)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                self.ln(2)
                continue
            for wl in textwrap.wrap(line, width=95) or [""]:
                self.set_x(L_MARGIN)
                self.multi_cell(EFF_W, 5, wl)
        self.ln(2)

    def shap_table(self, label: str, features: list[tuple[str, float]]):
        if not features:
            return
        self.set_font("Helvetica", "BI", 8)
        self.set_text_color(*LLOYDS_GREEN)
        self.set_x(L_MARGIN)
        self.cell(EFF_W, 5, f"  {label.capitalize()}", ln=True)
        self.set_text_color(*TEXT_DARK)
        col1, col2 = EFF_W * 0.75, EFF_W * 0.25
        for i, (fname, fval) in enumerate(features[:6]):
            fill_c = LLOYDS_LIGHT if i % 2 == 0 else WHITE
            self.set_fill_color(*fill_c)
            self.set_font("Helvetica", "", 8)
            self.set_x(L_MARGIN)
            self.cell(col1, 5, f"    {i+1}. {_sanitize(fname)}", fill=True)
            color = (0, 130, 60) if fval >= 0 else (180, 30, 30)
            self.set_text_color(*color)
            self.set_font("Helvetica", "B", 8)
            sign = "+" if fval >= 0 else "-"
            self.cell(col2, 5, f"{sign}{abs(fval):.4f}", fill=True, ln=True)
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
    pdf = _SMEReport(orientation="P", unit="mm", format="A4")
    pdf.set_margins(L_MARGIN, 20, R_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_text_color(*TEXT_DARK)

    # ── Cover title ────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*LLOYDS_GREEN)
    pdf.set_x(L_MARGIN)
    pdf.cell(EFF_W, 12, "SME Intelligence Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*TEXT_GREY)
    pdf.set_x(L_MARGIN)
    pdf.cell(EFF_W, 7, _sanitize(company_name), align="C", ln=True)
    pdf.ln(5)

    # ── Overview ───────────────────────────────────────────────────────────
    pdf.section_bar("COMPANY OVERVIEW")
    pdf.two_col("Sector",        sector,                                       shade=True)
    pdf.two_col("Report date",   datetime.now().strftime("%d %B %Y"),          shade=False)
    pdf.two_col("ML model",      "XGBoost (100k company sample)",              shade=True)
    pdf.two_col("Training data", "3,145,434 UK Companies House records",       shade=False)

    # ── Scores ─────────────────────────────────────────────────────────────
    pdf.section_bar("PREDICTION SCORES")
    # Header row
    pdf.set_fill_color(*LLOYDS_GREEN)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(L_MARGIN)
    pdf.cell(EFF_W * 0.50, 7, "  Signal",      fill=True)
    pdf.cell(EFF_W * 0.25, 7, "Probability",   fill=True)
    pdf.cell(EFF_W * 0.25, 7, "Status",        fill=True, ln=True)
    pdf.set_text_color(*TEXT_DARK)
    pdf.score_row("Growth Opportunity",  growth,  shade=True)
    pdf.score_row("Risk Signal",         risk,    shade=False)
    pdf.score_row("Lending Need Proxy",  lending, shade=True)

    # ── SHAP ──────────────────────────────────────────────────────────────
    if top_features and any(top_features.values()):
        pdf.section_bar("KEY DRIVERS (SHAP FEATURE IMPORTANCE)")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*TEXT_GREY)
        pdf.set_x(L_MARGIN)
        pdf.cell(EFF_W, 5,
                 "  Positive values (+) push prediction higher; negative (-) push it lower.",
                 ln=True)
        pdf.set_text_color(*TEXT_DARK)
        pdf.ln(2)
        for lbl in ["growth", "risk", "lending"]:
            feats = top_features.get(lbl, [])
            if feats:
                pdf.shap_table(lbl, feats)

    # ── AI narrative ───────────────────────────────────────────────────────
    pdf.section_bar("AI ANALYSIS")
    pdf.body_text(report_text)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    pdf.section_bar("DISCLAIMER")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*TEXT_GREY)
    pdf.set_x(L_MARGIN)
    pdf.multi_cell(EFF_W, 5,
        "This report is generated by machine learning models and a generative AI system. "
        "Predictions are probabilistic and should be used as decision-support tools only, "
        "not as the sole basis for lending or investment decisions. "
        "Data sourced from UK Companies House public bulk download. "
        "Lloyds Banking Group - Group 33 Dissertation Project.")
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
            sector=sector, growth=growth, risk=risk, lending=lending,
            report_text=report_text, top_features=top_features,
            company_name=company_name,
        )
        fname = (f"lloyds_sme_report_{sector.replace(' ', '_').lower()}"
                 f"_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            key=key,
        )
    except Exception as e:
        st.error(f"PDF generation error: {e}")
