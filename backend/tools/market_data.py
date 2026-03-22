import yfinance as yf
import json
import logging
from langchain.tools import tool
from tools.ticker_resolver import get_ticker
from functools import lru_cache
from utils.safe_execution import safe_execute_sync

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def fetch_yf_data(ticker_str: str) -> dict:
    """Caching wrapper for yfinance calls to prevent redundancies."""
    ticker = yf.Ticker(ticker_str)
    return ticker.info

def fetch_market_logic(company_name: str) -> str:
    """Core market fetch logic wrapped physically by safe wrapper."""
    logger.info(f"Fetching market data for: {company_name}")
    ticker_str = get_ticker(company_name)
    
    try:
        info = fetch_yf_data(ticker_str)
        
        # Adding error handling safely inside the try block
        market_stats = {
            "symbol": ticker_str,
            "companyName": info.get("longName", company_name),
            "currency": info.get("financialCurrency") or info.get("currency") or "USD",
            "currentPrice": info.get("currentPrice", "Data Not Available"),
            "marketCap": info.get("marketCap", "Data Not Available"),
            "revenue": info.get("totalRevenue", "Data Not Available"),
            "revenueGrowth": info.get("revenueGrowth", "Data Not Available"),
            "sector": info.get("sector", "Data Not Available"),
            "industry": info.get("industry", "Data Not Available"),
            "ebitda": info.get("ebitda", "Data Not Available"),
            "debtToEquity": info.get("debtToEquity", "Data Not Available")
        }
        
        return json.dumps(market_stats)
        
    except Exception as e:
        logger.error(f"Failed pulling market data natively for {company_name}: {e}")
        return json.dumps({
            "error": f"YFinance pull natively failed: {str(e)}",
            "source": "market_data"
        })

@tool
def get_market_data(company_name: str) -> str:
    """
    Useful to fetch real-time market data globally for a target company securely.
    Returns current price, market cap, revenue, revenue growth, and sector correctly wrapped in timeout resilience.
    """
    return safe_execute_sync(fetch_market_logic, "market_data", company_name)
