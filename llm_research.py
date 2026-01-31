"""
LLM Research Module — Perplexity Sonar Pro Integration

Performs grounded web research for top stock picks using Perplexity's Sonar Pro.
Replaces Gemini + Google Search grounding with better structure and source tracking.

Usage:
    from llm_research import research_asset
    result = research_asset("Toyota Motor", "7203.T", kind="jp_stock")
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Literal
from dataclasses import dataclass, asdict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the same directory as this script
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar-pro"
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_TEMPERATURE = 0.1
RETRY_DELAY_SECONDS = 2
MAX_RETRIES = 2


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResearchResult:
    """Structured output from LLM research."""
    recent_news_summary: str
    upcoming_catalysts: List[str]
    news_sentiment: Literal["Positive", "Negative", "Neutral", "Mixed"]
    key_risks: List[str]
    supporting_source_indices: List[int]
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    
    # Source tracking (stored separately, not in JSON output)
    search_results: Optional[List[Dict]] = None
    citations: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for scoring integration."""
        return {
            "recent_news_summary": self.recent_news_summary,
            "upcoming_catalysts": self.upcoming_catalysts,
            "news_sentiment": self.news_sentiment,
            "key_risks": self.key_risks,
            "catalyst_count": len(self.upcoming_catalysts),
            "risk_count": len(self.key_risks),
            "sentiment_score": self._sentiment_to_score(),
            "success": self.success,
        }
    
    def _sentiment_to_score(self) -> float:
        """Map sentiment to numeric score."""
        mapping = {
            "Positive": 1.0,
            "Mixed": 0.5,
            "Neutral": 0.2,
            "Negative": -1.0,
        }
        return mapping.get(self.news_sentiment, 0.0)


def _get_fallback_result(error_msg: str = "Insufficient data") -> ResearchResult:
    """Return safe fallback when research fails."""
    return ResearchResult(
        recent_news_summary="Research unavailable",
        upcoming_catalysts=[],
        news_sentiment="Neutral",
        key_risks=[],
        supporting_source_indices=[],
        success=False,
        error_message=error_msg,
    )


# =============================================================================
# JSON Schema for Perplexity Response Format
# =============================================================================

RESEARCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "recent_news_summary": {
            "type": "string",
            "description": "Concise summary of recent news within the lookback period"
        },
        "upcoming_catalysts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of concrete upcoming catalysts that could move price"
        },
        "news_sentiment": {
            "type": "string",
            "enum": ["Positive", "Negative", "Neutral", "Mixed"],
            "description": "Overall sentiment of recent news"
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key risks or negative factors"
        },
        "supporting_source_indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices referencing search_results that support this analysis"
        }
    },
    "required": ["recent_news_summary", "upcoming_catalysts", "news_sentiment"],
    "additionalProperties": False
}


# =============================================================================
# Prompt Templates
# =============================================================================

# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT_CRYPTO = """You are an expert crypto analyst. Use web search to find impactful news and upcoming catalysts that could move price. Prefer primary sources and major outlets. Focus on recent developments but include relevant older context if significant."""

SYSTEM_PROMPT_JP_STOCK = """You are an expert Japanese equities analyst. Use web search to find price-relevant news and upcoming catalysts. Prefer primary sources such as company releases, exchange filings (TDnet, EDINET), and reputable financial news (Nikkei, Reuters, Bloomberg). ALSO scan for social sentiment on platforms like X (Twitter) for retail buzz. Search in both Japanese and English. Focus on recent developments but include relevant older context if significant."""

USER_PROMPT_TEMPLATE_CRYPTO = """Analyze the cryptocurrency: {asset_name} ({ticker}).

Prioritize information from after {cutoff_date}, but do not strictly ignore important context from slightly before if it is still driving price action.

Focus on:
- Recent partnerships, exchange listings, funding rounds
- Token unlocks, airdrops, mainnet launches
- Regulatory news affecting this specific asset
- Whale movements or significant on-chain activity
- Social sentiment and trending narratives on X (Twitter)

Return a JSON object with these fields:
- recent_news_summary: string summarizing significant news (state dates clearly)
- upcoming_catalysts: array of strings, each a concrete upcoming event
- news_sentiment: one of "Positive", "Negative", "Neutral", "Mixed"
- key_risks: array of strings listing potential negative factors
- supporting_source_indices: array of integers referencing which search results you used

If no significant news exists, return empty arrays and neutral sentiment."""

USER_PROMPT_TEMPLATE_JP_STOCK = """Analyze the Japanese stock: {asset_name} ({ticker}).

Prioritize information from after {cutoff_date}, but do not strictly ignore important context from slightly before if it is still driving price action. Search in Japanese ("{asset_name} ニュース") and English.

Focus on:
- Earnings reports, guidance changes, dividend announcements
- M&A activity, major investments, asset sales
- Product launches, new contracts, expansion news
- Management changes, regulatory actions
- Social media buzz (X/Twitter) or retail investor sentiment (~yahoo~ finance boards)
- Any news that could cause a significant price move

Return a JSON object with these fields:
- recent_news_summary: string summarizing significant news (state dates clearly)
- upcoming_catalysts: array of strings, each a concrete upcoming event
- news_sentiment: one of "Positive", "Negative", "Neutral", "Mixed"
- key_risks: array of strings listing potential negative factors
- supporting_source_indices: array of integers referencing which search results you used

If no significant news exists, return empty arrays and neutral sentiment."""


# =============================================================================
# Core Research Function
# =============================================================================

def research_asset(
    asset_name: str,
    ticker: str,
    kind: Literal["crypto", "jp_stock"] = "jp_stock",
    lookback_days: int = 30,  # Increased to 30 days for better coverage
    api_key: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> ResearchResult:
    """
    Perform grounded web research on an asset using Perplexity Sonar Pro.
    
    Args:
        asset_name: Company or asset name (e.g., "Toyota Motor")
        ticker: Symbol (e.g., "7203.T" or "BTC")
        kind: "crypto" or "jp_stock" - affects prompt priorities
        lookback_days: Number of days to look back (default 30)
        api_key: Perplexity API key (uses PERPLEXITY_API_KEY env var if not provided)
        extra_context: Optional live context to include in the prompt (not from web search)
    
    Returns:
        ResearchResult with structured findings and source metadata
    """
    # Get API key
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY not found in environment")
        return _get_fallback_result("API key not configured")
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    cutoff_date_str = cutoff_date.strftime("%B %d, %Y")  # For prompt
    
    # Select prompts based on asset type
    if kind == "crypto":
        system_prompt = SYSTEM_PROMPT_CRYPTO
        user_prompt_template = USER_PROMPT_TEMPLATE_CRYPTO
    else:
        system_prompt = SYSTEM_PROMPT_JP_STOCK
        user_prompt_template = USER_PROMPT_TEMPLATE_JP_STOCK
    
    user_prompt = user_prompt_template.format(
        asset_name=asset_name,
        ticker=ticker,
        cutoff_date=cutoff_date_str,
    )
    if extra_context:
        extra_context = extra_context.strip()
        if extra_context:
            user_prompt += (
                "\n\nAdditional live context (provided by the trading system; may not appear in web search):\n"
                f"{extra_context}\n"
                "If this context is not provided, do not infer anything from its absence."
            )
    
    # Build request payload
    # NOTE: Removed search_after_date_filter to allow broader search context
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "web_search_options": {
            "search_context_size": "high"
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "research_output",
                "strict": True,
                "schema": RESEARCH_JSON_SCHEMA
            }
        },
    }
    
    # Add language filter for JP stocks
    if kind == "jp_stock":
        payload["search_language_filter"] = ["en", "ja"]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Make request with retry logic
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"Researching {asset_name} ({ticker}) - attempt {attempt + 1}")
            
            response = requests.post(
                PERPLEXITY_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # Extract content and metadata
            content = data["choices"][0]["message"]["content"]
            search_results = data.get("search_results", [])
            citations = data.get("citations", [])
            
            # Parse JSON response
            parsed = json.loads(content)
            
            # Build result object
            result = ResearchResult(
                recent_news_summary=parsed.get("recent_news_summary", ""),
                upcoming_catalysts=parsed.get("upcoming_catalysts", []),
                news_sentiment=parsed.get("news_sentiment", "Neutral"),
                key_risks=parsed.get("key_risks", []),
                supporting_source_indices=parsed.get("supporting_source_indices", []),
                success=True,
                search_results=search_results,
                citations=citations,
            )
            
            logger.info(f"Research complete for {ticker}: sentiment={result.news_sentiment}, catalysts={len(result.upcoming_catalysts)}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {ticker}: {e}")
            if attempt < MAX_RETRIES:
                # Retry with stricter prompt
                payload["messages"][1]["content"] += "\n\nIMPORTANT: Return ONLY valid JSON, no other text."
                payload["temperature"] = 0
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return _get_fallback_result(f"JSON parsing failed: {e}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {ticker}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
                continue
            return _get_fallback_result(f"API error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error researching {ticker}: {e}")
            return _get_fallback_result(f"Unexpected error: {e}")
    
    return _get_fallback_result("Max retries exceeded")


# =============================================================================
# Batch Research Function
# =============================================================================

def research_top_picks(
    picks: List[Dict],
    kind: Literal["crypto", "jp_stock"] = "jp_stock",
    delay_seconds: float = 2.0,
    max_picks: int = 20,
) -> List[Dict]:
    """
    Research multiple top picks with rate limiting.
    
    Args:
        picks: List of dicts with 'symbol' and optionally 'name' keys
        kind: Asset type for prompt selection
        delay_seconds: Delay between API calls
        max_picks: Maximum number of picks to research
    
    Returns:
        List of dicts with original pick data plus research results
    """
    results = []
    
    for i, pick in enumerate(picks[:max_picks]):
        symbol = pick.get("symbol", "")
        name = pick.get("name", symbol)  # Use symbol as name if not provided
        
        print(f"[{i+1}/{min(len(picks), max_picks)}] Researching {symbol}...")
        
        research = research_asset(
            asset_name=name,
            ticker=symbol,
            kind=kind,
        )
        
        # Combine pick data with research
        combined = {**pick, **research.to_dict()}
        combined["_research_metadata"] = {
            "search_results": research.search_results,
            "citations": research.citations,
        }
        
        results.append(combined)
        
        # Rate limiting
        if i < len(picks) - 1:
            time.sleep(delay_seconds)
    
    return results


# =============================================================================
# Convenience Functions for JP Stocks
# =============================================================================

def research_jp_stock(
    symbol: str,
    company_name: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> ResearchResult:
    """
    Convenience function for researching a single JP stock.
    
    Args:
        symbol: Stock symbol (e.g., "7203.T")
        company_name: Optional company name (if not provided, uses symbol)
    
    Returns:
        ResearchResult with findings
    """
    name = company_name or symbol.replace(".T", "")
    return research_asset(name, symbol, kind="jp_stock", extra_context=extra_context)


def research_crypto(ticker: str, coin_name: Optional[str] = None) -> ResearchResult:
    """
    Convenience function for researching a single cryptocurrency.
    
    Args:
        ticker: Crypto ticker (e.g., "BTC", "ETH")
        coin_name: Optional coin name (if not provided, uses ticker)
    
    Returns:
        ResearchResult with findings
    """
    name = coin_name or ticker
    return research_asset(name, ticker, kind="crypto")


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python llm_research.py <symbol> [company_name]")
        print("Example: python llm_research.py 7203.T 'Toyota Motor'")
        sys.exit(1)
    
    symbol = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else symbol
    
    print(f"\n🔍 Researching {name} ({symbol})...")
    print("=" * 60)
    
    result = research_jp_stock(symbol, name)
    
    print(f"\n📰 News Summary:")
    print(f"   {result.recent_news_summary}")
    
    print(f"\n📅 Upcoming Catalysts ({len(result.upcoming_catalysts)}):")
    for cat in result.upcoming_catalysts:
        print(f"   • {cat}")
    
    print(f"\n😊 Sentiment: {result.news_sentiment}")
    
    print(f"\n⚠️ Key Risks ({len(result.key_risks)}):")
    for risk in result.key_risks:
        print(f"   • {risk}")
    
    if result.citations:
        print(f"\n📚 Sources ({len(result.citations)}):")
        for i, cite in enumerate(result.citations[:5]):
            print(f"   [{i}] {cite}")
    
    print("\n" + "=" * 60)
