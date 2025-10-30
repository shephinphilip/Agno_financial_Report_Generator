# requirements:
# pip install agno pandas openpyxl python-docx PyMuPDF fpdf2 openai

import os
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
from fpdf import FPDF
from agno.models.openai import OpenAIChat
import re
from helper import read_file_to_df, _extract_pdf, _extract_docx
from agent import DataAssistantAgent, RiskAssessorAgent, MarketStrategistAgent


class StyledPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Add Liberation fonts if available, otherwise use built-in fonts
        try:
            font_regular = "liberation-fonts-ttf-2.1.5/LiberationSans-Italic.ttf"
            font_bold = "liberation-fonts-ttf-2.1.5/LiberationSans-Bold.ttf"

            if os.path.exists(font_regular) and os.path.exists(font_bold):
                self.add_font("LiberationSans", "", font_regular, uni=True)
                self.add_font("LiberationSans", "B", font_bold, uni=True)
                self.custom_font = True
            else:
                self.custom_font = False
        except:
            self.custom_font = False
    
    def header(self):
        if self.custom_font:
            self.set_font("LiberationSans", "B", 16)
        else:
            self.set_font("Arial", "B", 16)
        self.cell(0, 10, "FINANCIAL ANALYSIS REPORT", ln=True, align="C")
        self.ln(4)
        if self.custom_font:
            self.set_font("LiberationSans", "", 12)
        else:
            self.set_font("Arial", "", 12)
        self.cell(0, 5, "=" * 60, ln=True, align="C")
        self.ln(8)

    def section_title(self, title):
        if self.custom_font:
            self.set_font("LiberationSans", "B", 14)
        else:
            self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        self.ln(3)

    def section_body(self, text):
        if self.custom_font:
            self.set_font("LiberationSans", "", 11)
        else:
            self.set_font("Arial", "", 11)
        
        # Handle text encoding issues
        try:
            self.multi_cell(0, 7, text)
        except:
            # Fallback: remove problematic characters
            clean_text = text.encode('ascii', 'ignore').decode('ascii')
            self.multi_cell(0, 7, clean_text)
        self.ln(5)

def generate_pdf_report(text, output_path):
    pdf = StyledPDF()
    pdf.add_page()

    # Split the text into logical sections
    sections = text.split("\n")
    current_section = []
    title = None

    for line in sections:
        line_strip = line.strip()
        
        # Identify section headers (lines ending with ':' or all caps headers)
        if line_strip.endswith(":") and len(line_strip) > 3 and line_strip != "FINANCIAL ANALYSIS REPORT":
            if title and current_section:
                pdf.section_title(title)
                body_text = "\n".join(current_section)
                pdf.section_body(body_text)
                current_section = []
            title = line_strip
        elif line_strip == "FINANCIAL ANALYSIS REPORT" or line_strip.startswith("===="):
            continue  # Skip, already in header
        else:
            if line_strip:  # Only add non-empty lines
                current_section.append(line)

    # Add last section
    if title and current_section:
        pdf.section_title(title)
        body_text = "\n".join(current_section)
        pdf.section_body(body_text)

    pdf.output(output_path)
    return output_path


# === Orchestrator / Coordinator ===

class FinancialPipeline:
    def __init__(self, data_agent: DataAssistantAgent, risk_agent: RiskAssessorAgent, strat_agent: MarketStrategistAgent):
        self.data_agent = data_agent
        self.risk_agent = risk_agent
        self.strat_agent = strat_agent

    def execute(self, file_paths: list[str], report_path: str="financial_report.pdf"):
        print("\n" + "="*60)
        print("MULTI-AGENT FINANCIAL ANALYSIS SYSTEM")
        print("="*60)
        
        # Stage 1: Data ingestion
        print("\n[Stage 1] Data Processing Agent...")
        data_out = self.data_agent.run(file_paths)
        
        # Stage 2: Risk evaluation
        print("[Stage 2] Risk Assessment Agent...")
        risk_in = {
            "dataframe": data_out["dataframe"],
            "data_comments": self._safe_text(data_out["llm_comments"])
        }
        risk_out = self.risk_agent.run(risk_in)
        
        # Stage 3: Market strategy
        print("[Stage 3] Market Strategy Agent...")
        strat_in = {
            "risk_score": risk_out["risk_score"],
            "risk_narrative": self._safe_text(risk_out["llm_risk_narrative"]),
            "data_insights": self._safe_text(data_out["llm_comments"])
        }
        strat_out = self.strat_agent.run(strat_in)

        # Compose final report text
        print("[Stage 4] Generating PDF Report...")
        report_lines = []
        report_lines.append("FINANCIAL ANALYSIS REPORT")
        report_lines.append("="*60)
        report_lines.append("")
        
        report_lines.append("Data Processing Summary:")
        report_lines.append(data_out["raw_summary"])
        report_lines.append("")
        report_lines.append("Data Analysis Insights:")
        report_lines.append(self._safe_text(data_out["llm_comments"]))
        report_lines.append("")
        
        report_lines.append("Risk Evaluation:")
        report_lines.append(f"Calculated Risk Score: {risk_out['risk_score']}/100")
        report_lines.append("")
        report_lines.append("Risk Assessment Details:")
        report_lines.append(self._safe_text(risk_out["llm_risk_narrative"]))
        report_lines.append("")
        
        report_lines.append("Market Strategy Recommendations:")
        report_lines.append(self._safe_text(strat_out["llm_strategy"]))
        report_lines.append("")

        # Clean markdown formatting
        cleaned_lines = [self._strip_markdown(line) for line in report_lines]
        full_text = "\n".join(cleaned_lines)
        
        # Generate PDF
        output = generate_pdf_report(full_text, report_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Analysis Complete!")
        print(f"✓ Report generated: {report_path}")
        print(f"{'='*60}\n")
        
        return output
    
    def _safe_text(self, obj):
        """Extract readable text whether it's a string or RunResponse"""
        if hasattr(obj, "content"):
            return obj.content
        return str(obj)
    
    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting for PDF"""
        if not isinstance(text, str):
            return str(text)
        # Remove bold/italic markers, headers, and list markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)      # bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)          # italic
        text = re.sub(r"#+\s*", "", text)                 # markdown headers
        text = re.sub(r"^[-•]\s*", "", text, flags=re.M)  # bullet dashes
        text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code
        return text.strip()


# === Example Usage ===

if __name__ == "__main__":
    files = [
        "data/finance.csv"
    ]
    pipeline = FinancialPipeline(DataAssistantAgent(), RiskAssessorAgent(), MarketStrategistAgent())
    report_file = pipeline.execute(files, report_path="final_financial_report.pdf")
    print("Report available at:", report_file)