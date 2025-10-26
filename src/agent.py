"""LangChain agent implementation for SME analysis using FAISS and Ollama.

This file implements:
- SMEAgent: LangChain agent with tools for financial analysis
- Tool functions: profit calculation, period summary, and business recommendations
"""
from typing import List, Tuple, Any
import os
import re
import calendar
import pandas as pd
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from src.utils import EmbeddingHelper, format_currency
from src.vectorstore import FAISSStore


class SMEAgent:
    def __init__(self, persist_dir: str = "faiss_store", llm_model: str = "llama2"):
        """Initialize LangChain agent with FAISS store, Ollama LLM, and business analysis tools.
        
        Args:
            persist_dir: Directory containing FAISS index and metadata
            llm_model: Name of Ollama model to use (e.g., 'llama2', 'mistral', 'codellama')
        """
        self.persist_dir = persist_dir
        try:
            self.store = FAISSStore.load(persist_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS store from '{persist_dir}'. Run ingestion first: {str(e)}")

        self.embedder = EmbeddingHelper()
        
        self.llm = OllamaLLM(model=llm_model)
        
        # Initialize tools
        self.tools = [
            Tool(
                name="search_financials",
                func=self.retrieve_context,
                description="Search through financial records and business data to find relevant information. Input should be a specific question about finances or business metrics."
            ),
            Tool(
                name="calculate_profit",
                func=self.calculate_monthly_profit,
                description="Calculate profit for a specific month. Input should be a month in format 'MMM-YY' like 'Jan-23'."
            ),
            Tool(
                name="analyze_period",
                func=self.analyze_period,
                description="Analyze financial metrics for multiple months. Input should be months separated by commas like 'Jan-23,Feb-23,Mar-23'."
            )
        ]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert business analyst assistant. Your responses must:
1. Be based ONLY on the exact financial data shown in the provided records
2. Never make up or modify any numbers
3. Answer questions directly using the actual figures shown"""),
            HumanMessage(content="{question}")
        ])
        
        # Create the chain
        self.chain = prompt | self.llm | StrOutputParser()

    def retrieve_context(self, query: str, k: int = 10) -> str:
        """Tool for retrieving relevant financial context.
        
        Args:
            query: The search query
            k: Number of results to retrieve
            
        Returns:
            Formatted string with relevant financial records
        """
        q_emb = self.embedder.embed([query])[0]
        results = self.store.search([q_emb], n_results=k)
        context = []
        
        # Extract month from query if present
        import re
        month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}', query)
        query_month = month_match.group(0) if month_match else None
        
        # Convert Q1, Q2, etc. to month ranges
        quarter_match = re.search(r'Q(\d) (\d{4})', query)
        if quarter_match:
            q = int(quarter_match.group(1))
            year = quarter_match.group(2)[2:]  # Last 2 digits
            start_month = ((q - 1) * 3) + 1
            months = [f"{calendar.month_abbr[m]}-{year}" for m in range(start_month, start_month + 3)]
            query_months = months
        else:
            query_months = [query_month] if query_month else None
            
            # Add header with request context
        if query_months:
            context.append(f"Based on our financial records for {', '.join(query_months)}:\n")        # Collect and sort records
        records = []
        for doc in results['documents'][0]:
            # If specific months requested, only include those
            if query_months:
                if any(month in doc for month in query_months):
                    records.append(doc)
            else:
                records.append(doc)
        
        # Sort chronologically
        def extract_month(record):
            match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})', record)
            if not match:
                return "zzz"  # Put non-matching at end
            month, year = match.groups()
            month_num = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
                        "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}[month]
            return f"20{year}{month_num:02d}"
            
        records.sort(key=extract_month)
        
        if not records:
            context.append("No matching financial records found.")
        else:
            context.extend(records)
            
            # Add summary for multiple months
            if len(records) > 1:
                context.append("\nSummary Metrics:")
                
                # Extract numeric values
                sales = []
                expenses = []
                profits = []
                customers = []
                inventory = []
                marketing = []
                
                for record in records:
                    for line in record.split('\n'):
                        if 'Sales Revenue' in line:
                            sales.append(float(re.search(r'₹([\d,]+)', line).group(1).replace(',','')))
                        elif 'Total Expenses' in line:
                            expenses.append(float(re.search(r'₹([\d,]+)', line).group(1).replace(',','')))
                        elif 'Profit:' in line:
                            profits.append(float(re.search(r'₹([\d,]+)', line).group(1).replace(',','')))
                        elif 'Active Customers' in line:
                            customers.append(float(re.search(r': (\d+)', line).group(1)))
                        elif 'Inventory Cost' in line:
                            inventory.append(float(re.search(r'₹([\d,]+)', line).group(1).replace(',','')))
                        elif 'Marketing Spend' in line:
                            marketing.append(float(re.search(r'₹([\d,]+)', line).group(1).replace(',','')))
                
                # Calculate period metrics
                if sales:
                    context.append(f"- Average Monthly Revenue: ₹{sum(sales)/len(sales):,.0f}")
                    context.append(f"- Total Period Revenue: ₹{sum(sales):,.0f}")
                if expenses:
                    context.append(f"- Average Monthly Expenses: ₹{sum(expenses)/len(expenses):,.0f}")
                    context.append(f"- Total Period Expenses: ₹{sum(expenses):,.0f}")
                if profits:
                    context.append(f"- Average Monthly Profit: ₹{sum(profits)/len(profits):,.0f}")
                    context.append(f"- Total Period Profit: ₹{sum(profits):,.0f}")
                    context.append(f"- Overall Profit Margin: {(sum(profits)/sum(sales))*100:.1f}%")
                if customers:
                    context.append(f"- Average Monthly Customers: {sum(customers)/len(customers):.0f}")
                if marketing:
                    context.append(f"- Total Marketing Spend: ₹{sum(marketing):,.0f}")
                    context.append(f"- Marketing ROI: {(sum(profits)/sum(marketing))*100:.1f}%")
                
        return "\n".join(context)

    def calculate_monthly_profit(self, month: str) -> str:
        """Tool for calculating profit for a specific month.
        
        Args:
            month: Month in format 'MMM-YY' like 'Jan-23'
            
        Returns:
            Formatted profit calculation with context
        """
        context = self.retrieve_context(f"financial data for {month}")
        rows = [line for line in context.split('\n') if month in line]
        if not rows:
            return f"No data found for month {month}"
            
        try:
            for row in rows:
                if 'Sales' in row and 'Expenses' in row:
                    # Extract sales and expenses
                    sales = float([x for x in row.split() if 'Sales' in x][0].replace(',', ''))
                    expenses = float([x for x in row.split() if 'Expenses' in x][0].replace(',', ''))
                    profit = sales - expenses
                    return f"""
Month: {month}
Sales: {format_currency(sales)}
Expenses: {format_currency(expenses)}
Profit: {format_currency(profit)}
"""
            return f"Could not parse financial data for {month}"
        except Exception as e:
            return f"Error calculating profit: {str(e)}"

    def analyze_period(self, months_str: str) -> str:
        """Tool for analyzing multiple months of financial data.
        
        Args:
            months_str: Comma-separated months in format 'MMM-YY'
            
        Returns:
            Period analysis with trends and metrics
        """
        months = [m.strip() for m in months_str.split(',')]
        results = []
        total_sales = 0
        total_expenses = 0
        
        for month in months:
            context = self.retrieve_context(f"financial data for {month}")
            rows = [line for line in context.split('\n') if month in line]
            
            if rows:
                for row in rows:
                    if 'Sales' in row and 'Expenses' in row:
                        sales = float([x for x in row.split() if 'Sales' in x][0].replace(',', ''))
                        expenses = float([x for x in row.split() if 'Expenses' in x][0].replace(',', ''))
                        profit = sales - expenses
                        total_sales += sales
                        total_expenses += expenses
                        results.append(f"{month}: Profit {format_currency(profit)} (Sales {format_currency(sales)} - Expenses {format_currency(expenses)})")
        
        if not results:
            return "No data found for the specified months"
            
        period_profit = total_sales - total_expenses
        summary = f"""Period Analysis ({len(months)} months):
Total Sales: {format_currency(total_sales)}
Total Expenses: {format_currency(total_expenses)}
Total Profit: {format_currency(period_profit)}
Average Monthly Profit: {format_currency(period_profit/len(months))}

Monthly Breakdown:
""" + "\n".join(results)
        
        return summary

    def answer_query(self, query: str) -> str:
        """Process a business query using retrieved context and LLM.
        
        Args:
            query: User's business analysis question
            
        Returns:
            LLM response with analysis and recommendations
        """
        try:
            # Get relevant context
            context = self.retrieve_context(query)
            
            # Debug output
            print("\nDEBUG - Context provided to LLM:")
            print("-" * 40)
            print(context)
            print("-" * 40)
            
            # Format the request more explicitly
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a financial analyst. Your task is to answer questions about business performance using ONLY the financial records shown below.
                
IMPORTANT: 
1. Use the EXACT numbers from the records
2. Do not make up or modify any values
3. Answer the specific question asked
4. Reference actual data in your response"""),
                HumanMessage(content=f"Here are the financial records to analyze:\n\n{context}\n\nBased on these records, please answer this question: {query}")
            ])
            
            # Create a new chain for this query
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    # Alias for consistency
    analyze = answer_query


def _load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df


def profit_in_month(csv_path: str, month: str) -> str:
    df = _load_csv(csv_path)
    row = df[df['Month'] == month]
    if row.empty:
        return f"Month {month} not found in dataset."
    sales = int(row['Sales (INR)'].values[0])
    expenses = int(row['Expenses (INR)'].values[0])
    profit = sales - expenses
    return f"Profit in {month} = {format_currency(profit)} (Sales {format_currency(sales)} - Expenses {format_currency(expenses)})"


def summarize_period(csv_path: str, months: List[str]) -> str:
    df = _load_csv(csv_path)
    sel = df[df['Month'].isin(months)]
    if sel.empty:
        return "No matching months found."
    total_sales = int(sel['Sales (INR)'].sum())
    total_expenses = int(sel['Expenses (INR)'].sum())
    avg_customers = int(sel['Customers'].mean())
    net_profit = total_sales - total_expenses
    return (
        f"Total Sales: {format_currency(total_sales)}\n"
        f"Total Expenses: {format_currency(total_expenses)}\n"
        f"Average Customers: {avg_customers}\n"
        f"Net Profit: {format_currency(net_profit)}"
    )


def recommend_for_month(csv_path: str, month: str, n: int = 2) -> List[str]:
    df = _load_csv(csv_path)
    if month not in df['Month'].values:
        return [f"Month {month} not found."]
    idx = df.index[df['Month'] == month][0]
    start = max(0, idx - 5)
    window = df.iloc[start:idx+1]

    recs = []
    max_inv = window['Inventory Cost (INR)'].max()
    inv_for_month = int(df.loc[idx, 'Inventory Cost (INR)'])
    if inv_for_month >= max_inv:
        recs.append(f"Reduce inventory holding costs: ₹{inv_for_month:,d} in {month} is high compared to recent months.")
    else:
        recs.append("Inventory costs are within recent range; keep monitoring.")

    min_mark = window['Marketing Spend (INR)'].min()
    mark_for_month = int(df.loc[idx, 'Marketing Spend (INR)'])
    if mark_for_month <= min_mark:
        recs.append(f"Consider increasing marketing spend: ₹{mark_for_month:,d} in {month} is low compared to recent months and may affect customer acquisition.")
    else:
        recs.append("Marketing spend is at/above recent levels. Consider targeting/spending efficiency improvements.")

    return recs