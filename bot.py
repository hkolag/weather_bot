import logging
import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, List
import sys

# Console logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Verified NWS point URLs (official grid for Central Park NYC and Miami Airport)
NWS_FORECAST_URLS = {
    'Miami': 'https://api.weather.gov/gridpoints/MFL/109,69/forecast',  # Miami
    'NYC': 'https://api.weather.gov/gridpoints/OKX/33,35/forecast'      # Central Park area
}

CITIES = ['Miami', 'NYC']
SERIES_TICKERS = {'Miami': 'KXHIGHMIA', 'NYC': 'KXHIGHNY'}  # Confirmed from Kalshi data
SIGMAS = {'Miami': 1.4, 'NYC': 1.6}
ACCURACIES = {'Miami': 0.976, 'NYC': 0.952}

PUBLIC_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

BANKROLL = 50.0
MAX_RISK_PER_TRADE_PCT = 0.04
MAX_TRADES_PER_CITY = 2

def fetch_nws_forecast(city: str) -> float:
    url = NWS_FORECAST_URLS[city]
    headers = {'User-Agent': 'KalshiBot/1.0 (contact@example.com)'}
    try:
        resp = requests.get(url, headers=headers, timeout=10).json()
        periods = resp['properties']['periods']
        tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
        for period in periods:
            start = datetime.fromisoformat(period['startTime'].rstrip('Z') + '+00:00')
            if period['isDaytime'] and start.date() == tomorrow:
                mu = float(period['temperature'])
                logger.info(f"{city} NWS high forecast: {mu}°F")
                return mu
        logger.warning(f"No tomorrow daytime forecast for {city}")
        return 78.0 if city == 'Miami' else 32.0  # Current winter defaults
    except Exception as e:
        logger.error(f"NWS error for {city}: {e}")
        return 78.0 if city == 'Miami' else 32.0

def fetch_kalshi_markets(series_ticker: str) -> Dict[Tuple[float, float], float]:
    url = f"{PUBLIC_KALSHI_BASE}/markets?series_ticker={series_ticker}&status=open&limit=100"
    try:
        resp = requests.get(url, timeout=10).json()
        markets = resp.get('markets', [])
        if not markets:
            logger.warning(f"No open markets for {series_ticker}")
            return {}
        
        probs = {}
        for market in markets:
            subtitle = market.get('subtitle', '').replace('°F', '').replace('°', '').strip().lower()
            yes_bid = market.get('yes_bid', 0) / 100.0
            if not subtitle:
                continue
            
            if 'to' in subtitle:
                parts = subtitle.split('to')
                low = float(parts[0].strip())
                high = float(parts[1].strip()) + 1
                probs[(low, high)] = yes_bid
            elif 'or above' in subtitle:
                low = float(subtitle.split('or above')[0].strip())
                probs[(low, 200)] = yes_bid
            elif 'or below' in subtitle:
                high = float(subtitle.split('or below')[0].strip()) + 1
                probs[( -50, high )] = yes_bid  # Arbitrary low
        
        logger.info(f"Fetched {len(probs)} bins for {series_ticker}")
        return probs
    except Exception as e:
        logger.error(f"Kalshi error for {series_ticker}: {e}")
        return {}

# compute_edges and main same as previous full script (copy them here)

# ... (keep the same compute_edges and main functions from the last version I gave)

if __name__ == "__main__":
    main()
