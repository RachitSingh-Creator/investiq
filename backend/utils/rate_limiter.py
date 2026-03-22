import time
from fastapi import Request, HTTPException
import logging
from collections import defaultdict
from utils.config import settings

logger = logging.getLogger(__name__)

# Basic dictionary acting natively as a local LRU store mapping client IPs safely
_request_counts = defaultdict(list)

async def check_rate_limit(request: Request):
    """Dependency injecting strict threshold checking matching standard sliding bounds."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    current_time = time.time()
    
    # Prune elements strictly older than 60 seconds
    _request_counts[client_ip] = [
        t for t in _request_counts[client_ip] 
        if current_time - t < 60
    ]
    
    if len(_request_counts[client_ip]) >= settings.rate_limit_per_minute:
        logger.warning(f"Rate limit exceeded structurally by explicitly detected IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded cleanly. Max {settings.rate_limit_per_minute} requests natively permitted per minute securely."
        )
        
    _request_counts[client_ip].append(current_time)
