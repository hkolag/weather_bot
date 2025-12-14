import os
import logging
import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, List

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config: High-predictability cities only
CITIES = ['Miami', 'NYC']
ACCURACIES = {'Miami': 0.976, 'NYC': 0.952}
SIGMAS = {'Miami': 1.4, 'NYC': 1.6}  # High temp SDs
COORDS_LAT_LON = {'Miami': (25.7617, -80.1918), 'NYC': (40.7128, -74.0060)}

KALSHI_BASE = 'https://trading-api.kalshi.com/trade-api/v2'
NWS_BASE = 'https://api.weather.gov'
KALSHI_API_KEY = os.getenv('KALSHI_API_KEY')  # Set in Render env vars

BANKROLL = 50.0
MAX_RISK_PER_TRADE_PCT = 0.04  # 4% = $2 max risk
MAX_TRADES_PER_CITY = 2

def fetch_nws_forecast(city: str) -> float:
    lat, lon = COORDS_LAT_LON[city]
    url = f"{NWS_BASE}/points/{lat},{lon}/forecast"
    try:
        resp = requests.get(url, headers={'User-Agent': 'KalshiBot/1.0'}).json()
        periods = resp['properties']['periods']
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        for period in periods:
            if period['isDaytime'] and datetime.fromisoformat(period['startTime'].replace('Z', '+00:00')).date() == tomorrow:
                mu = float(period['temperature'])
                logger.info(f"Fetched {city} high forecast: {mu}°F")
                return mu
        logger.warning(f"No daytime forecast found for {city} tomorrow")
        return 75.0 if city == 'Miami' else 45.0  # Reasonable fallback
    except Exception as e:
        logger.error(f"NWS error for {city}: {e}")
        return 75.0 if city == 'Miami' else 45.0

def fetch_kalshi_markets(city: str) -> Dict[Tuple[float, float], float]:
    city_lower = city.lower().replace('nyc', 'ny')
    series_ticker = f"HIGH-{city_lower.upper()}"
    url = f"{KALSHI_BASE}/markets?event_ticker_prefix={series_ticker}&status=open"
    headers = {'Authorization': f'Bearer {KALSHI_API_KEY}'} if KALSHI_API_KEY else {}
    try:
        resp = requests.get(url, headers=headers).json()
        markets = resp.get('markets', [])
        if not markets:
            logger.warning(f"No open high temp markets for {city}")
            return {}
        
        # Take the next-day market (usually first or by close_time)
        market = min(markets, key=lambda m: m['close_time'])  # Closest closing
        orderbook_url = f"{KALSHI_BASE}/markets/{market['ticker']}/orderbook"
        ob_resp = requests.get(orderbook_url, headers=headers).json()
        
        probs = {}
        # Parse ticks – Kalshi uses tick-based orderbook; approximate from yes_bid/ask
        yes_bids = ob_resp['orderbook'].get('yes_bid', [])
        for bid in yes_bids:
            # Simplified: assume bins from market subtitles or standard 2°F
            # In practice, parse market['subtitle'] or ticks; fallback to common structure
            # For robustness, you'd map from market range data – here simulate typical
            logger.info(f"Fetched market {market['ticker']} for {city}")
        # Placeholder – improve with full tick parsing in production
        return {(70,72): 0.10, (72,74): 0.30}  # REPLACE with real parsing
    except Exception as e:
        logger.error(f"Kalshi error for {city}: {e}")
        return {}

def compute_edges(mu: float, sigma: float, market_probs: Dict, accuracy: float) -> List[dict]:
    threshold = 0.055 / accuracy  # ~4.5% Miami, 5.5% NYC
    edges = []
    for (low, high), market_yes_p in market_probs.items():
        model_yes_p = norm.cdf(high - 0.5, mu, sigma) - norm.cdf(low - 0.5, mu, sigma)
        market_no_p = 1 - market_yes_p
        model_no_p = 1 - model_yes_p
        
        diff_no = model_no_p - market_no_p
        diff_yes = model_yes_p - market_yes_p
        
        # Prioritize No on tails
        if low > mu + sigma and diff_no > threshold + 0.01:  # Upper tail boost
            edge, action, price = diff_no, "Buy No", market_no_p
        elif diff_yes > threshold + 0.02:  # Rare Yes
            edge, action, price = diff_yes, "Buy Yes", market_yes_p
        else:
            continue
        
        kelly = (edge ** 2) / (price if action == "Buy Yes" else (1 - price))
        risk = min(kelly * BANKROLL, BANKROLL * MAX_RISK_PER_TRADE_PCT)
        contracts = max(1, int(risk / price)) if price > 0 else 0
        
        edges.append({
            'Bin': f"{int(low)}-{int(high-1)}°F",
            'Action': action,
            'Edge': round(edge * 100, 1),
            'Risk': round(risk, 2),
            'Contracts': contracts,
            'Price': round(price * 100, 1)
        })
    
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    return edges[:MAX_TRADES_PER_CITY]  # Top 1-2 only

def main():
    logger.info("=== Kalshi High Temp Bot Run Started ===")
    total_risk = 0.0
    for city in CITIES:
        mu = fetch_nws_forecast(city)
        sigma = SIGMAS[city]
        market_probs = fetch_kalshi_markets(city)
        if not market_probs:
            continue
        accuracy = ACCURACIES[city]
        edges = compute_edges(mu, sigma, market_probs, accuracy)
        
        for edge in edges:
            entry = f"{city} High | {edge['Bin']} | {edge['Action']} @ {edge['Price']}¢ | Edge: {edge['Edge']}% | Risk: ${edge['Risk']} | Contracts: {edge['Contracts']}"
            logger.info(entry)
            total_risk += edge['Risk']
    
    if total_risk > 0:
        logger.info(f"Total risked today: ${total_risk:.2f}")
    else:
        logger.info("No high-conviction edges today – standing down.")
    logger.info("=== Bot Run Ended ===")

if __name__ == "__main__":
    main()
