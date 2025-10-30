from agno.models.openai import OpenAIChat
from helper import read_file_to_df, _extract_pdf, _extract_docx
import pandas as pd
import os
from agno.agent import Agent
from config import OPENAI_API_KEY

# === Define Agents ===

class DataAssistantAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
            description="Agent that reads multiple financial data files and normalizes into a dataframe.",
            markdown=False
        )

    def run(self, file_paths: list[str]) -> dict:
        # Step 1: read data
        dfs = []
        for p in file_paths:
            print(f"Reading file: {p}")
            dfs.append(read_file_to_df(p))
        
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        summary = f"Read {len(file_paths)} files. Combined dataframe has shape {combined.shape}."
        
        # Get basic statistics if numeric columns exist
        numeric_cols = combined.select_dtypes(include=['number']).columns.tolist()
        stats_summary = ""
        if numeric_cols:
            stats = combined[numeric_cols].describe().to_string()
            stats_summary = f"\nNumeric columns statistics:\n{stats}"
        
        # Optionally let the LLM comment on data anomalies
        prompt = f"""Here is a description of the financial data:
        - Combined dataframe has shape {combined.shape}
        - Columns: {list(combined.columns)}
        {stats_summary}
        
        Provide a brief analysis noting:
        1. Key observations about the data structure
        2. Any notable patterns or anomalies
        3. Data quality assessment
        
        Keep response concise and professional."""
        
        response = self.agent.run(prompt)
        
        return {
            "dataframe": combined,
            "raw_summary": summary,
            "llm_comments": response
        }

class RiskAssessorAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
            description="Agent that evaluates financial risks (liquidity, credit, market) given a data frame.",
            markdown=False
        )

    def run(self, input_data: dict) -> dict:
        df = input_data["dataframe"]
        data_comments = input_data.get("data_comments", "")
        
        # Calculate basic risk metrics
        count = len(df)
        risk_score = min(100, round(count * 0.1, 2))
        
        # Analyze numeric columns for volatility if available
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        volatility_info = ""
        if numeric_cols:
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                std = df[col].std()
                mean = df[col].mean()
                if mean != 0:
                    cv = (std / abs(mean)) * 100  # Coefficient of variation
                    volatility_info += f"\n- {col}: CV = {cv:.2f}%"
        
        prompt = f"""Conduct a comprehensive risk assessment based on:
        
        Data Overview:
        - Dataframe length: {count} records
        - Initial risk score: {risk_score}/100
        {volatility_info}
        
        Data Analyst Comments:
        {data_comments}
        
        Provide a professional risk assessment covering:
        1. Overall risk rating (Low/Medium/High)
        2. Specific risk factors identified
        3. Liquidity and volatility concerns
        4. Recommended risk mitigation strategies
        
        Keep response structured and actionable."""
        
        response = self.agent.run(prompt)
        
        return {
            "risk_score": risk_score,
            "llm_risk_narrative": response
        }

class MarketStrategistAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
            description="Agent that formulates a market strategy based on risk assessment and data analysis.",
            markdown=False
        )

    def run(self, input_data: dict) -> dict:
        score = input_data["risk_score"]
        risk_narrative = input_data.get("risk_narrative", "")
        data_insights = input_data.get("data_insights", "")
        
        prompt = f"""Develop a comprehensive market strategy based on:
        
        Risk Score: {score}/100
        
        Risk Assessment:
        {risk_narrative}
        
        Data Insights:
        {data_insights}
        
        Provide strategic recommendations including:
        1. Overall strategic direction
        2. Investment priorities and allocations
        3. Growth opportunities identified
        4. Specific action items with timeline
        5. Key performance indicators to monitor
        
        Keep recommendations practical and actionable."""
        
        response = self.agent.run(prompt)
        
        return {
            "llm_strategy": response
        }
