import logging
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import norm
from typing import Dict, List
import sys
import os
import json
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
SIGMAS = {'Miami': 1.5, 'NYC': 1.8}
ACCURACIES = {'Miami': 0.98, 'NYC': 0.96}

PUBLIC_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'
TRADING_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

BANKROLL = 160
MAX_RISK_PER_TRADE_PCT = 0.04
MAX_TRADES_PER_CITY = 4

# Minimum No price to trade (60 cents = 0.60)
MIN_NO_PRICE = 0.70

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
        resp = requests.get(url, headers=headers, timeout=30).json()
        periods = resp['properties']['periods']
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)
        tomorrow = (now_eastern + timedelta(days=1)).date()
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
        resp = requests.get(url, timeout=30).json()
        markets = resp.get('markets', [])
        if not markets:
            logger.warning(f"No open markets found for {series_ticker}")
            return {}
        
        # Use Eastern Time for "tomorrow"
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)
        tomorrow = (now_eastern + timedelta(days=1)).date()
        tomorrow_str = tomorrow.strftime("%y%b%d").upper()
        
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
    private_key = serialization.load_pem_private_key(
        KALSHI_PRIVATE_KEY_PEM.encode(), 
        password=None
    )
    # Correct full path including /trade-api/v2
    path = "/trade-api/v2/portfolio/orders"
    message = f"{timestamp}POST{path}".encode('utf-8')
    
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH  # Match Kalshi docs exactly
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
        if resp.status_code in [200, 201]:
            order = resp.json().get('order', {})
            logger.info(f"SUCCESS: Placed {contracts} {side} on {ticker} @ {price_cents}¢ | Order ID: {order.get('order_id')}")
        else:
            logger.error(f"Order failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Order exception: {e}")

def compute_edges(mu: float, sigma: float, market_probs: Dict, accuracy: float, city: str) -> List[dict]:
    base_threshold = 0.055 / accuracy
    no_threshold = base_threshold - 0.015 if city == 'NYC' else base_threshold
    cold_boost = 0.02 if city == 'NYC' and mu < 40 else 0
    
    edges = []
    
    for key, value in market_probs.items():
        low, high = key
        market_yes_p, ticker = value
        if market_yes_p <= 0.01 or market_yes_p >= 0.99:
            continue
        
        market_no_p = 1 - market_yes_p
        
        # NEW: Skip if No price < 60¢
        if market_no_p < MIN_NO_PRICE:
            continue
        
        model_yes_p = norm.cdf(high - 0.5, mu, sigma) - norm.cdf(low - 0.5, mu, sigma)
        model_no_p = 1 - model_yes_p
        
        diff_no = model_no_p - market_no_p
        
        action = None
        edge_val = 0.0
        price = 0.0
        
        if diff_no > no_threshold - cold_boost:
            action = "Buy No"
            edge_val = diff_no
            price = market_no_p
        elif ENABLE_YES_BUYS and model_yes_p - market_yes_p > base_threshold + 0.05:
            action = "Buy Yes"
            edge_val = model_yes_p - market_yes_p
            price = market_yes_p
        
        if action:
            denominator = price if action == "Buy Yes" else (1 - price)
            if denominator == 0:
                continue
            kelly = edge_val ** 2 / denominator
            risk = min(kelly * BANKROLL, BANKROLL * MAX_RISK_PER_TRADE_PCT)
            contracts = max(1, int(risk / price)) if price > 0 else 0
            
            potential_profit = contracts * (1 - price) if action == "Buy No" else contracts * (1 - price)
            
            bin_str = f">={int(low)}°F" if high >= 200 else f"<={int(high-1)}°F" if low <= -100 else f"{int(low)}-{int(high-1)}°F"
            
            edges.append({
                'Bin': bin_str,
                'Action': action,
                'Edge': round(edge_val * 100, 1),
                'Price': round(price * 100),
                'Risk': round(risk, 2),
                'Contracts': contracts,
                'PotentialProfit': round(potential_profit, 2),
                'Ticker': ticker
            })
    
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    return edges[:MAX_TRADES_PER_CITY]

def main():
    logger.info("=== Kalshi High Temp Bot Started ===")
    logger.info(f"ENABLE_YES_BUYS: {ENABLE_YES_BUYS} | ENABLE_AUTO_TRADING: {ENABLE_AUTO_TRADING}")
    total_risk = 0.0
    total_potential_profit = 0.0
    trade_count = 0
    
    for city in CITIES:
        series = SERIES_TICKERS[city]
        logger.info(f"Processing {city} high temp...")
        mu = fetch_nws_forecast(city)
        sigma = SIGMAS[city]
        market_probs = fetch_kalshi_markets(series)
        
        if not market_probs:
            logger.info(f"No market data available for {city} – skipping")
            continue
        
        edges = compute_edges(mu, sigma, market_probs, ACCURACIES[city], city)
        
        for edge in edges:
            entry = (f"{city} High | {edge['Bin']} | {edge['Action']} @ {edge['Price']}¢ | "
                     f"Edge: {edge['Edge']}% | Risk: ${edge['Risk']} | Contracts: {edge['Contracts']}")
            logger.info(entry)
            total_risk += edge['Risk']
            total_potential_profit += edge['PotentialProfit']
            trade_count += 1
            
            # Real auto-trading
            if edge['Ticker']:
                side = "no" if edge['Action'] == "Buy No" else "yes"
                place_order(edge['Ticker'], side, edge['Contracts'], edge['Price'])
    
    if trade_count > 0:
        logger.info(f"Recommended {trade_count} trades | Total risked: ${total_risk:.2f} | Total potential profit: ${total_potential_profit:.2f} (if all win)")
    else:
        logger.info("No high-conviction edges found today – standing down.")
    
    logger.info("=== Bot Run Ended ===\n")

if __name__ == "__main__":
    main()
