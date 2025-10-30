"""
helper.py
==========
Utility module for reading and extracting financial data from various file formats.

Supported formats:
    - CSV (.csv)
    - Excel (.xls, .xlsx)
    - PDF (.pdf)
    - Word Document (.docx)

Each supported file is parsed into a pandas DataFrame, ensuring downstream
agents in the financial analysis pipeline receive uniform data structures.
"""

import os
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from typing import Any


# ======================================================================
# Main File Reader
# ======================================================================

# === Helper functions for file reading ===

def read_file_to_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def _extract_pdf(path: str) -> pd.DataFrame:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    full = "\n".join(texts)
    return pd.DataFrame({"content": [full]})

def _extract_docx(path: str) -> pd.DataFrame:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full = "\n".join(paragraphs)
    return pd.DataFrame({"content": [full]})


def prepare_financial_data(self, file_paths: list[str]) -> str:
    """Read and prepare financial data with enhanced summary"""
    print("\n[Preparing Data] Reading financial files...")
    
    dfs = []
    file_summaries = []
    
    for path in file_paths:
        try:
            df = read_file_to_df(path)
            dfs.append(df)
            
            # Create summary for this file
            file_name = os.path.basename(path)
            summary = f"\nFile: {file_name}\n"
            summary += f"Shape: {df.shape}\n"
            
            if 'content' in df.columns:
                # Text-based file
                summary += f"Type: Text document\n"
                summary += f"Preview: {df['content'].iloc[0][:300]}...\n"
            else:
                # Structured data
                summary += f"Columns: {', '.join(df.columns.tolist())}\n"
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    summary += f"Numeric columns: {', '.join(numeric_cols)}\n"
                    summary += f"\nStatistics:\n{df[numeric_cols].describe().to_string()}\n"
            
            file_summaries.append(summary)
            print(f"  ✓ Processed: {file_name}")
            
        except Exception as e:
            print(f"  ✗ Error reading {path}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    # **ENHANCED**: Calculate key metrics upfront to help agents
    numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
    
    key_metrics = ""
    if 'Totals.Revenue' in combined_df.columns and 'Totals.Expenditure' in combined_df.columns:
        avg_revenue = combined_df['Totals.Revenue'].mean()
        avg_expenditure = combined_df['Totals.Expenditure'].mean()
        deficit_pct = ((avg_expenditure - avg_revenue) / avg_revenue * 100)
        
        key_metrics = f"""
CALCULATED KEY METRICS:
- Average Revenue: ${avg_revenue:,.0f} (in thousands)
- Average Expenditure: ${avg_expenditure:,.0f} (in thousands)
- Average Deficit/Surplus: {deficit_pct:.1f}% ({'DEFICIT' if deficit_pct > 0 else 'SURPLUS'})
"""
        
        if 'Totals. Debt at end of fiscal year' in combined_df.columns:
            avg_debt = combined_df['Totals. Debt at end of fiscal year'].mean()
            debt_to_revenue = avg_debt / avg_revenue
            key_metrics += f"- Average Debt: ${avg_debt:,.0f} (in thousands)\n"
            key_metrics += f"- Debt-to-Revenue Ratio: {debt_to_revenue:.2f}\n"
        
        if 'Details.Education.Education Total' in combined_df.columns:
            avg_education = combined_df['Details.Education.Education Total'].mean()
            education_pct = (avg_education / avg_expenditure * 100)
            key_metrics += f"- Education Spending: ${avg_education:,.0f} ({education_pct:.1f}% of expenditure)\n"
    
    # Format data for agents
    data_context = f"""
FINANCIAL DATA OVERVIEW:
========================
Total files processed: {len(file_paths)}
Combined dataset shape: {combined_df.shape}
Number of records: {len(combined_df)}
{key_metrics}

DETAILED FILE ANALYSIS:
{''.join(file_summaries)}
"""
    return data_context