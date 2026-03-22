import requests
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_ticker(company_name: str) -> str:
    """
    Resolve a company name into a correct stock ticker by querying the Yahoo Finance Search API.
    Utilizes LRU caching to reduce redundant network requests.
    """
    logger.info(f"Resolving ticker for company: {company_name}")
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        quotes = data.get('quotes', [])
        if quotes:
            # Prefer equities
            equity_quotes = [q for q in quotes if q.get('quoteType') == 'EQUITY']
            if equity_quotes:
                resolved_ticker = equity_quotes[0]['symbol']
                logger.info(f"Resolved {company_name} to {resolved_ticker} (Equity)")
                return resolved_ticker
            
            # Fallback to the first available if no equity found
            resolved_ticker = quotes[0]['symbol']
            logger.info(f"Resolved {company_name} to {resolved_ticker} (Non-Equity)")
            return resolved_ticker
            
    except requests.RequestException as e:
        logger.error(f"Error resolving ticker for {company_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when resolving ticker for {company_name}: {e}")
        
    logger.warning(f"Using fallback string for ticker: {company_name}")
    return company_name
