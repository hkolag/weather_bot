import os
import logging
import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, List
import sys

# Console-only logging (Render captures all stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration
CITIES = {
    'Miami': 'KXHIGHMIA',
    'NYC': 'KXHIGHNY'
}
SIGMAS = {'Miami': 1.4, 'NYC': 1.6}
ACCURACIES = {'Miami': 0.976, 'NYC': 0.952}
COORDS = {'Miami': (25.7617, -80.1918), 'NYC': (40.7128, -74.0060)}

PUBLIC_KALSHI_BASE = 'https://api.elections.kalshi.com/trade-api/v2'
NWS_BASE = 'https://api.weather.gov'

BANKROLL = 50.0
MAX_RISK_PER_TRADE_PCT = 0.04  # 4% = $2 max risk per trade
MAX_TRADES_PER_CITY = 2

def fetch_nws_forecast(city: str) -> float:
    lat, lon = COORDS[city]
    url = f"{NWS_BASE}/points/{lat},{lon}/forecast"
    headers = {'User-Agent': 'KalshiWeatherBot/1.0 (contact@example.com)'}
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
        logger.warning(f"No daytime forecast found for {city} tomorrow")
        return 78.0 if city == 'Miami' else 35.0
    except Exception as e:
        logger.error(f"NWS fetch error for {city}: {e}")
        return 78.0 if city == 'Miami' else 35.0

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
            subtitle = market.get('subtitle', '').replace('°F', '').strip()
            yes_bid = market.get('yes_bid', 0) / 100.0  # Decimal probability
            
            if not subtitle:
                continue
            
            if '-' in subtitle and subtitle.replace('-', '').replace('.', '').isdigit():
                parts = subtitle.split('-')
                low = float(parts[0])
                high = float(parts[1]) + 1  # e.g., 78-79 → (78, 80)
                probs[(low, high)] = yes_bid
            elif '<=' in subtitle:
                low = float(subtitle.split('<=')[1])
                probs[(0, low + 1)] = yes_bid
            elif '>=' in subtitle:
                high = float(subtitle.split('>=')[1])
                probs[(high, 200)] = yes_bid  # Arbitrary high cap
        
        logger.info(f"Fetched {len(probs)} bins for {series_ticker}")
        return probs
    except Exception as e:
        logger.error(f"Kalshi fetch error for {series_ticker}: {e}")
        return {}

def compute_edges(mu: float, sigma: float, market_probs: Dict[Tuple[float, float], float], accuracy: float) -> List[dict]:
    threshold = 0.055 / accuracy  # ~4.5% Miami, ~5.5% NYC
    edges = []
    
    for (low, high), market_yes_p in market_probs.items():
        model_yes_p = norm.cdf(high - 0.5, mu, sigma) - norm.cdf(low - 0.5, mu, sigma)
        market_no_p = 1 - market_yes_p
        model_no_p = 1 - model_yes_p
        
        diff_no = model_no_p - market_no_p
        diff_yes = model_yes_p - market_yes_p
        
        action = None
        edge = 0.0
        price = 0.0
        
        # Prioritize Buy No on upper tails
        if low > mu + sigma and diff_no > threshold + 0.01:
            action, edge, price = "Buy No", diff_no, market_no_p
        elif diff_yes > threshold + 0.02:  # Rare strong Yes
            action, edge, price = "Buy Yes", diff_yes, market_yes_p
        
        if action:
            kelly = (edge ** 2) / (price if action == "Buy Yes" else (1 - price))
            risk = min(kelly * BANKROLL, BANKROLL * MAX_RISK_PER_TRADE_PCT)
            contracts = max(1, int(risk / price)) if price > 0 else 0
            
            edges.append({
                'Bin': f"{int(low)}-{int(high-1)}°F" if high < 200 else f">={int(low)}°F",
                'Action': action,
                'Edge': round(edge * 100, 1),
                'Price': round(price * 100),
                'Risk': round(risk, 2),
                'Contracts': contracts
            })
    
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    return edges[:MAX_TRADES_PER_CITY]

def main():
    logger.info("=== Kalshi High Temp Bot Started ===")
    total_risk = 0.0
    trades = 0
    
    for city, series in CITIES.items():
        logger.info(f"Processing {city} high temp market...")
        mu = fetch_nws_forecast(city)
        sigma = SIGMAS[city]
        market_probs = fetch_kalshi_markets(series)
        
        if not market_probs:
            logger.info(f"No market data for {city} – skipping")
            continue
        
        accuracy = ACCURACIES[city]
        edges = compute_edges(mu, sigma, market_probs, accuracy)
        
        for edge in edges:
            entry = (f"{city} High | {edge['Bin']} | {edge['Action']} @ {edge['Price']}¢ | "
                     f"Edge: {edge['Edge']}% | Risk: ${edge['Risk']} | Contracts: {edge['Contracts']}")
            logger.info(entry)
            total_risk += edge['Risk']
            trades += 1
    
    if trades > 0:
        logger.info(f"Total trades recommended: {trades} | Total risked: ${total_risk:.2f}")
    else:
        logger.info("No high-conviction edges today – standing down.")
    
    logger.info("=== Bot Run Ended ===\n")

if __name__ == "__main__":
    main()
