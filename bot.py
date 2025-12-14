import logging
import requests
from datetime import datetime, timedelta, timezone
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

BANKROLL = 50.0
MAX_RISK_PER_TRADE_PCT = 0.04  # $2 max risk
MAX_TRADES_PER_CITY = 2

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

def fetch_kalshi_markets(series_ticker: str) -> Dict[Tuple[float, float], float]:
    url = f"{PUBLIC_KALSHI_BASE}/markets?series_ticker={series_ticker}&status=open&limit=100"
    try:
        resp = requests.get(url, timeout=15).json()
        markets = resp.get('markets', [])
        if not markets:
            logger.warning(f"No open markets found for {series_ticker}")
            return {}
        
        probs = {}
        for market in markets:
            subtitle = market.get('subtitle', '').replace('°F', '').replace('°', '').strip()
            yes_bid = market.get('yes_bid', 0) / 100.0
            if not subtitle:
                continue
            
            subtitle_lower = subtitle.lower()
            if ' to ' in subtitle_lower:
                parts = subtitle_lower.split(' to ')
                low = float(parts[0].strip())
                high = float(parts[1].strip()) + 1
                probs[(low, high)] = yes_bid
            elif 'or above' in subtitle_lower:
                low_str = subtitle_lower.split('or above')[0].strip()
                low = float(low_str)
                probs[(low, 200.0)] = yes_bid
            elif 'or below' in subtitle_lower:
                high_str = subtitle_lower.split('or below')[0].strip()
                high = float(high_str) + 1
                probs[(-100.0, high)] = yes_bid  # Wider low for cold tails
        
        logger.info(f"Fetched {len(probs)} price bins for {series_ticker}")
        return probs
    except Exception as e:
        logger.error(f"Kalshi fetch error for {series_ticker}: {e}")
        return {}

def compute_edges(mu: float, sigma: float, market_probs: Dict[Tuple[float, float], float], accuracy: float) -> List[dict]:
    threshold = 0.055 / accuracy  # ~4.5% Miami, ~5.5% NYC
    edges = []
    
    for (low, high), market_yes_p in market_probs.items():
        if market_yes_p >= 0.99 or market_yes_p <= 0.01:
            continue  # Skip truly illiquid/near-certain bins (protects from 0¢/99¢ issues)
        
        model_yes_p = norm.cdf(high - 0.5, mu, sigma) - norm.cdf(low - 0.5, mu, sigma)
        market_no_p = 1 - market_yes_p
        model_no_p = 1 - model_yes_p
        
        diff_no = model_no_p - market_no_p
        diff_yes = model_yes_p - market_yes_p
        
        action = None
        edge_val = 0.0
        price = 0.0
        
        # Prioritize Buy No on any overpriced tail (cold or hot)
        if abs(diff_no) > abs(diff_yes) and diff_no > threshold:
            action = "Buy No"
            edge_val = diff_no
            price = market_no_p
        elif diff_yes > threshold + 0.02:  # Strong Yes on undervalued core/shoulder
            action = "Buy Yes"
            edge_val = diff_yes
            price = market_yes_p
        
        if action:
            denominator = price if action == "Buy Yes" else (1 - price)
            kelly = edge_val ** 2 / denominator
            risk = min(kelly * BANKROLL, BANKROLL * MAX_RISK_PER_TRADE_PCT)
            contracts = max(1, int(risk / price)) if price > 0 else 0
            
            if high >= 200:
                bin_str = f">={int(low)}°F"
            elif low <= -100:
                bin_str = f"<={int(high-1)}°F"
            else:
                bin_str = f"{int(low)}-{int(high-1)}°F"
            
            edges.append({
                'Bin': bin_str,
                'Action': action,
                'Edge': round(edge_val * 100, 1),
                'Price': round(price * 100),
                'Risk': round(risk, 2),
                'Contracts': contracts
            })
    
    edges.sort(key=lambda x: x['Edge'], reverse=True)
    return edges[:MAX_TRADES_PER_CITY]

def main():
    logger.info("=== Kalshi High Temp Bot Started ===")
    total_risk = 0.0
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
        
        edges = compute_edges(mu, sigma, market_probs, ACCURACIES[city])
        
        for edge in edges:
            entry = (f"{city} High | {edge['Bin']} | {edge['Action']} @ {edge['Price']}¢ | "
                     f"Edge: {edge['Edge']}% | Risk: ${edge['Risk']} | Contracts: {edge['Contracts']}")
            logger.info(entry)
            total_risk += edge['Risk']
            trade_count += 1
    
    if trade_count > 0:
        logger.info(f"Recommended {trade_count} trades | Total risked: ${total_risk:.2f}")
    else:
        logger.info("No high-conviction edges found today – standing down.")
    
    logger.info("=== Bot Run Ended ===\n")

if __name__ == "__main__":
    main()
