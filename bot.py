import logging
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.stats import norm
from typing import Dict, List
import sys
import os
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Console logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Verified NWS grid forecast URLs
NWS_FORECAST_URLS = {
    'Miami': 'https://api.weather.gov/gridpoints/MFL/109,69/forecast',
    'NYC': 'https://api.weather.gov/gridpoints/OKX/33,35/forecast'
}

CITIES = ['Miami', 'NYC']
SERIES_TICKERS = {'Miami': 'KXHIGHMIA', 'NYC': 'KXHIGHNY'}
SIGMAS = {'Miami': 1.4, 'NYC': 1.6}
ACCURACIES = {'Miami': 0.976, 'NYC': 0.952}

PUBLIC_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'
TRADING_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

BANKROLL = 50.0
MAX_RISK_PER_TRADE_PCT = 0.04  # $2 max risk
MAX_TRADES_PER_CITY = 2

# Triggers
ENABLE_YES_BUYS = os.getenv('ENABLE_YES_BUYS', 'false').lower() == 'true'
ENABLE_AUTO_TRADING = os.getenv('ENABLE_AUTO_TRADING', 'false').lower() == 'true'
KALSHI_API_KEY_ID = os.getenv('KALSHI_API_KEY_ID')
KALSHI_PRIVATE_KEY_PEM = os.getenv('KALSHI_PRIVATE_KEY_PEM')

if ENABLE_AUTO_TRADING:
    if not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PEM:
        logger.warning("Auto-trading enabled but missing API credentials — falling back to logging only")
        ENABLE_AUTO_TRADING = False
    else:
        logger.info("Auto-trading is ENABLED — real orders will be placed")
else:
    logger.info("Auto-trading is OFF — recommendations only")

def fetch_nws_forecast(city: str) -> float:
    url = NWS_FORECAST_URLS[city]
    headers = {'User-Agent': 'KalshiWeatherBot/1.0 (contact@example.com)'}
    try:
        resp = requests.get(url, headers=headers, timeout=15).json()
        periods = resp['properties']['periods']
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        for period in periods:
            start_str = period['startTime'].replace('Z', '+00:00')
            start = datetime.fromisoformat(start_str)
            if period['isDaytime'] and start.date() == tomorrow:
                mu = float(period['temperature'])
                logger.info(f"{city} NWS high forecast tomorrow: {mu}°F")
                return mu
        logger.warning(f"No tomorrow daytime period found for {city}")
        return 76.0 if city == 'Miami' else 32.0
    except Exception as e:
        logger.error(f"NWS error for {city}: {e}")
        return 76.0 if city == 'Miami' else 32.0

def fetch_kalshi_markets(series_ticker: str) -> Dict:
    url = f"{PUBLIC_KALSHI_BASE}/markets?series_ticker={series_ticker}&status=open&limit=100"
    try:
        resp = requests.get(url, timeout=15).json()
        markets = resp.get('markets', [])
        if not markets:
            logger.warning(f"No open markets found for {series_ticker}")
            return {}
        
        # Filter to tomorrow's event
        tomorrow_str = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%y%b%d").upper()
        tomorrow_markets = [m for m in markets if tomorrow_str in m.get('event_ticker', '')]
        
        logger.info(f"Found {len(tomorrow_markets)} markets for tomorrow ({tomorrow_str}) out of {len(markets)} open")
        if not tomorrow_markets:
            logger.warning("No tomorrow's markets found")
            return {}
        
        probs = {}
        for market in tomorrow_markets:
            subtitle = market.get('subtitle', '').replace('°F', '').replace('°', '').strip()
            yes_bid = market.get('yes_bid', 0) / 100.0
            ticker = market.get('ticker', '')
            if not subtitle or not ticker:
                continue
            
            subtitle_lower = subtitle.lower()
            if ' to ' in subtitle_lower:
                parts = subtitle_lower.split(' to ')
                low = float(parts[0].strip())
                high = float(parts[1].strip()) + 1
                probs[(low, high)] = (yes_bid, ticker)
            elif 'or above' in subtitle_lower:
                low = float(subtitle_lower.split('or above')[0].strip())
                probs[(low, 200.0)] = (yes_bid, ticker)
            elif 'or below' in subtitle_lower:
                high = float(subtitle_lower.split('or below')[0].strip()) + 1
                probs[(-100.0, high)] = (yes_bid, ticker)
        
        logger.info(f"Parsed {len(probs)} price bins for tomorrow's market")
        return probs
    except Exception as e:
        logger.error(f"Kalshi fetch error for {series_ticker}: {e}")
        return {}

def sign_payload(timestamp: str) -> str:
    private_key = serialization.load_pem_private_key(KALSHI_PRIVATE_KEY_PEM.encode(), password=None)
    message = f"{timestamp}POST/portfolio/orders"
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode()

def place_order(ticker: str, side: str, contracts: int, price_cents: int):
    if not ENABLE_AUTO_TRADING:
        logger.info(f"AUTO-TRADING OFF — would place: {contracts} {side} on {ticker} @ {price_cents}¢")
        return
    
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    payload = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": contracts,
        "type": "limit",
        "client_order_id": f"bot-{timestamp}",
    }
    if side == "yes":
        payload["yes_price"] = price_cents
    else:
        payload["no_price"] = price_cents
    
    signature = sign_payload(timestamp)
    
    headers = {
        "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    url = f"{TRADING_KALSHI_BASE}/portfolio/orders"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            order = resp.json().get('order', {})
            logger.info(f"SUCCESS: Placed {contracts} {side} on {ticker} @ {price_cents}¢ | Order ID: {order.get('order_id')}")
        else:
            logger.error(f"Order failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Order exception: {e}")

# compute_edges and main (same as previous final version)

if __name__ == "__main__":
    main()
