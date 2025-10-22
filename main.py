"""
AI Agent for Commerce Website - Palona Take-Home Exercise

This is a single AI agent that handles:
1. General conversation
2. Text-based product recommendations
3. Image-based product search

Reference: Amazon Rufus
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
from io import BytesIO
import asyncio
import re
import time

# Lightweight HTTP client for fetching page metadata (for images)
try:
    import httpx  # comes with perplexity dependency
    HTTPX_AVAILABLE = True
except Exception:
    HTTPX_AVAILABLE = False

from flask import Flask, request, jsonify, render_template_string

# Optional CORS support
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, tool

# Optional BLIP captioning
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False

# Perplexity imports
try:
    from perplexity import Perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    print("Warning: Perplexity not available. Using basic search.")

# Environment setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyByF353130s7tt7hmFrkN6eLha0t2JyNzw")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "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")
USE_GEMINI_AGENT = os.environ.get("USE_GEMINI_AGENT", "true").lower() == "true"
# Smalltalk controls
USE_LLM_FOR_CHITCHAT = os.environ.get("USE_LLM_FOR_CHITCHAT", "true").lower() == "true"
USE_PERPLEXITY_FOR_CHITCHAT = os.environ.get("USE_PERPLEXITY_FOR_CHITCHAT", "true").lower() == "true"

# Simple in-memory caches and circuit breaker
search_cache: Dict[str, Dict[str, Any]] = {}
image_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes
GEMINI_BLOCK_UNTIL = 0.0

# Load product catalog
with open('product_catalog.json', 'r') as f:
    PRODUCT_CATALOG = json.load(f)

# Simple keyword-based search (no embeddings API needed - avoids quota issues)
def keyword_search_products(query: str, k: int = 3):
    """
    Search products using simple keyword matching.
    This doesn't require embeddings API quota.
    """
    query_lower = query.lower()
    scored_products = []
    
    for product in PRODUCT_CATALOG:
        score = 0
        searchable_text = f"{product['name']} {product['category']} {product['description']} {' '.join(product['features'])} {' '.join(product['tags'])}".lower()
        
        # Count keyword matches
        for word in query_lower.split():
            if len(word) > 2:  # Ignore very short words
                if word in searchable_text:
                    score += searchable_text.count(word)
        
        if score > 0:
            scored_products.append((score, product))
    
    # Sort by score and return top k
    scored_products.sort(reverse=True, key=lambda x: x[0])
    return [product for score, product in scored_products[:k]]

print(" Product catalog loaded successfully with keyword search!")

# --- Intent detection helpers for better routing ---
def is_product_intent(message: str) -> bool:
    product_keywords = [
        'recommend', 'search', 'find', 'looking for', 'need', 'want to buy', 'shopping',
        'product', 'item', 'show me', 'buy', 'purchase', 'price',
        'cheap', 'expensive', 'best', 'good', 'quality', 'review', 'compare'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in product_keywords)

def is_greeting_or_chitchat(message: str) -> bool:
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', "what's up", 'how is it going', 'nice to meet you',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you'
    ]
    message_lower = message.lower().strip()
    return any(message_lower.startswith(g) or message_lower == g for g in greetings)

def is_knowledge_question(message: str) -> bool:
    indicators = ['what is', 'how does', 'why', 'when', 'where', 'explain', 'tell me about']
    message_lower = message.lower()
    return any(ind in message_lower for ind in indicators)

# --- Helper: extract a clean, product-oriented query from the image analysis ---
def extract_product_query_from_image_description(description: str) -> str:
    if not description:
        return ""
    text = description.strip()
    # 1) Prefer a concise description if present
    m = re.search(r"Concise Description[^:]*:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Concise Description[^:]*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip().strip('"')
    # 2) Look for 'Search Terms' list; take first 2–3 terms
    terms = []
    st = re.search(r"Search\s*Terms[^:]*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if st:
        raw = st.group(1)
        # split on * or ,
        parts = re.split(r"\*|,|;|\|", raw)
        for p in parts:
            t = p.strip().strip('-').strip()
            if len(t) > 2:
                terms.append(t)
        if terms:
            return ", ".join(terms[:3])
    # 3) Build query from attributes like Color/Style/Type/Brand
    attrs = []
    for key in ["Color", "Style", "Type", "Brand", "Category"]:
        m = re.search(rf"{key}\s*:\s*([^*\n]+)", text, flags=re.IGNORECASE)
        if m:
            attrs.append(m.group(1).strip())
    if attrs:
        return " ".join(attrs)
    # 4) Fallback to the original text (Perplexity wrapper will add shopping intent)
    return text[:200]

# --- Helper: Try to fetch Open Graph/Twitter image from a URL ---
def try_fetch_page_image(target_url: str) -> Optional[str]:
    if not HTTPX_AVAILABLE:
        return None
    try:
        with httpx.Client(follow_redirects=True, timeout=1.0, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PalonaBot/1.0)"
        }) as client:
            # Prefer HEAD, fallback to GET if needed
            resp = client.get(target_url)
            if resp.status_code >= 400:
                return None
            html = resp.text
            # Look for common meta tags
            patterns = [
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+name=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+name=["\']twitter:image:src["\'][^>]+content=["\']([^"\']+)["\']',
            ]
            for pat in patterns:
                m = re.search(pat, html, flags=re.IGNORECASE)
                if m:
                    img = m.group(1).strip()
                    # Basic validation
                    if img.startswith('http'):
                        return img
            # Fallback to site favicon (higher success rate than empty)
            try:
                from urllib.parse import urlparse
                net = urlparse(target_url)
                if net.netloc:
                    return f"https://www.google.com/s2/favicons?domain={net.netloc}&sz=128"
            except Exception:
                pass
            return None
    except Exception:
        return None

# --- BLIP captioning (loaded lazily on first use) ---
_blip_model = None
_blip_processor = None

def load_blip_if_needed():
    global _blip_model, _blip_processor
    if not BLIP_AVAILABLE:
        return False
    if _blip_model is None or _blip_processor is None:
        try:
            _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            return True
        except Exception as e:
            print(f"BLIP load error: {e}")
            return False
    return True

def caption_image_bytes(image_bytes: bytes) -> Optional[str]:
    if not load_blip_if_needed():
        return None
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        inputs = _blip_processor(images=image, return_tensors="pt")
        out = _blip_model.generate( inputs, max_new_tokens=30)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"BLIP caption error: {e}")
        return None

# Enhanced search using Perplexity API for real-time product search
def perplexity_product_search(query: str, search_type: str = "shopping") -> List[Dict[str, Any]]:
    """
    Use Perplexity's Search API to find real products online with images, prices, and links.
    This replaces local catalog search with live web results.
    """
    if not PERPLEXITY_AVAILABLE or not PERPLEXITY_API_KEY:
        print("Warning: Perplexity API not available. Falling back to local catalog.")
        return keyword_search_products(query, k=5)
        
    # Cache lookup
    key = f"{search_type}|{query.strip().lower()}"
    now = time.time()
    cached = search_cache.get(key)
    if cached and now - cached["ts"] < CACHE_TTL_SECONDS:
        return cached["data"]

    try:
        client = Perplexity(api_key=PERPLEXITY_API_KEY)
        
        # Create shopping-focused query
        if search_type == "shopping":
            enhanced_query = f"{query} buy online shopping price"
        else:
            enhanced_query = query
        
        # Perplexity SDK 0.17.0 does not support a 'country' parameter.
        # Keep params compatible with installed version.
        search = client.search.create(
            query=enhanced_query,
            max_results=5,
            max_tokens_per_page=512
        )
        
        products = []
        
        for i, result in enumerate(search.results):
            # Parse product information from search results
            snippet = result.snippet
            title = result.title
            url = result.url
            
            # Extract price if mentioned
            import re
            price_match = re.search(r'\$(\d+(?:\.\d{2})?)', snippet)
            price = float(price_match.group(1)) if price_match else 0.0
            
            # Try to get a real product image from the page (Open Graph/Twitter)
            cached_img = image_cache.get(url)
            image_url = None
            if cached_img and now - cached_img["ts"] < CACHE_TTL_SECONDS:
                image_url = cached_img["url"]
            if not image_url:
                image_url = try_fetch_page_image(url)
                if image_url:
                    image_cache[url] = {"url": image_url, "ts": now}
            if not image_url:
                # Fallback placeholder
                safe_title = re.sub(r"[^A-Za-z0-9+]+", "+", title)[:20]
                image_url = f"https://via.placeholder.com/300x300?text={safe_title}"
            
            # Create product-like structure
            product = {
                "id": f"perplexity_{i}",
                "name": title[:100],  # Limit title length
                "price": price if price > 0 else "Check website",
                "category": "Online Product",
                "description": snippet[:200] + "...",
                "features": ["Available Online", "Real-time Pricing"],
                "colors": ["Various"],
                "sizes": ["Various"],
                "image_url": image_url,
                "link_url": url
            }
            
            products.append(product)
        
        products = products[:5]
        search_cache[key] = {"ts": now, "data": products}
        return products
        
    except Exception as e:
        print(f"Perplexity search error: {e}")
        # Fallback to local catalog
        return keyword_search_products(query, k=5)


# --- Smalltalk / General conversation helpers ---
def smalltalk_llm_reply(message: str, chat_history: List = None) -> Optional[str]:
    """Use Gemini LLM to produce a natural conversational reply."""
    try:
        if chat_history is None:
            chat_history = []
        conversation_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            google_api_key=GEMINI_API_KEY
        )
        system_prompt = (
            "You are Palona, a friendly, concise shopping assistant. "
            "Chat naturally, ask brief clarifying questions when helpful. "
            "If the user seems to want products, you may say you can search real online stores."
        )
        messages = [SystemMessage(content=system_prompt)]
        if chat_history:
            messages.extend(chat_history[-6:])
        messages.append(HumanMessage(content=message))
        response = conversation_llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Smalltalk LLM error: {e}")
        return None

def smalltalk_perplexity_reply(message: str) -> Optional[str]:
    """Use Perplexity for knowledge-style smalltalk replies."""
    if not PERPLEXITY_AVAILABLE or not PERPLEXITY_API_KEY:
        return None
    try:
        client = Perplexity(api_key=PERPLEXITY_API_KEY)
        search = client.search.create(
            query=f"Answer conversationally: {message}",
            max_results=2,
            max_tokens_per_page=256
        )
        if search.results:
            result = search.results[0]
            snippet = (result.snippet or "").strip()
            if snippet:
                return snippet[:600]
    except Exception as e:
        print(f"Smalltalk Perplexity error: {e}")
    return None

def get_smalltalk_response(message: str, chat_history: List = None) -> str:
    """Select best smalltalk answer from LLM, Perplexity, or deterministic fallback."""
    if USE_LLM_FOR_CHITCHAT:
        ans = smalltalk_llm_reply(message, chat_history)
        if ans:
            return ans
    if USE_PERPLEXITY_FOR_CHITCHAT and is_knowledge_question(message):
        ans = smalltalk_perplexity_reply(message)
        if ans:
            return ans
    if is_greeting_or_chitchat(message):
        return "Hi! I'm Palona. I can chat and also help you find products. What's up?"
    return (
        "I'm here to chat and help with shopping. Ask me anything, or tell me "
        "what you're looking to buy and I can search real stores."
    )

# Tools for the AI Agent
@tool
def search_products_by_text(query: str) -> str:
    """
    Search for products online using Perplexity API.
    Returns real products with actual prices, images, and shopping links.
    
    Args:
        query: The user's search query or description of what they're looking for
    
    Returns:
        HTML formatted product results with images and links
    """
    # Use Perplexity for real-time online product search
    matched_products = perplexity_product_search(query, search_type="shopping")
    
    if not matched_products:
        return "No products found matching your query. Please try a different search term."
    
    # Format response with HTML for rich product cards
    response = f"<div style='margin-bottom: 15px; color: #666;'>🔍 Found {len(matched_products)} product(s) from online stores:</div>\n\n"
    
    # Add note about real-time results
    response += "<div style='background: #e8f4fd; border-radius: 8px; padding: 10px; margin-bottom: 15px; font-size: 0.9em; color: #1976d2;'>"
    response += "These are real products from online retailers with current pricing and availability. Click the links to view on the actual store."
    response += "</div>\n\n"
    
    for product in matched_products:
        # Create HTML product card with image and link
        price_display = f"${product['price']}" if isinstance(product['price'], (int, float)) else product['price']
        
        product_html = f"""
<div style="background: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="display: flex; gap: 15px;">
        <div style="flex-shrink: 0;">
            <img src="{product.get('image_url', 'https://via.placeholder.com/300?text=Image')}" 
                 loading="lazy" decoding="async"
                 style="width: 120px; height: 120px; object-fit: contain; background: #fff; border-radius: 8px; border: 2px solid #ddd;"
                 alt="{product['name']}" 
                 onerror="this.src='https://via.placeholder.com/100?text=No+Image'">
        </div>
        <div style="flex: 1;">
            <h3 style="margin: 0 0 8px 0; color: #333; font-size: 1.1em;">
                <a href="{product.get('link_url', '#')}" target="_blank" style="color: #667eea; text-decoration: none; hover: underline;">
                    {product['name']} 🔗
                </a>
            </h3>
            <p style="margin: 0 0 8px 0; color: #667eea; font-weight: bold; font-size: 1.3em;">
                {price_display}
            </p>
            
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.95em;">
                <strong>Description:</strong> {product['description']}
            </p>
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.9em;">
                <strong>Features:</strong> {', '.join(product.get('features', ['Check product page']))}
            </p>
            <div style="margin-top: 10px;">
                <a href="{product.get('link_url', '#')}" target="_blank" 
                   style="background: #667eea; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; display: inline-block; font-size: 0.9em;">
                    View on Store →
                </a>
            </div>
        </div>
    </div>
</div>
"""
        response += product_html
    
    return response


@tool
def get_product_by_id(product_id: str) -> str:
    """
    Get detailed information about a specific product by its ID.
    
    Args:
        product_id: The product ID to look up
    
    Returns:
        Detailed product information
    """
    product = next((p for p in PRODUCT_CATALOG if p['id'] == product_id), None)
    
    if not product:
        return f"Product with ID {product_id} not found."
    
    response = f"""
<div style="background: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea;">
    <div style="display: flex; gap: 15px;">
        <div style="flex-shrink: 0;">
            <img src="{product.get('image_url', 'https://via.placeholder.com/100')}" 
                 style="width: 100px; height: 100px; object-fit: cover; border-radius: 8px; border: 2px solid #ddd;"
                 alt="{product['name']}" 
                 onerror="this.src='https://via.placeholder.com/100?text=No+Image'">
        </div>
        <div style="flex: 1;">
            <h3 style="margin: 0 0 8px 0; color: #333; font-size: 1.1em;">
                <a href="{product.get('link_url', '#')}" target="_blank" style="color: #667eea; text-decoration: none;">
                    {product['name']}
                </a>
            </h3>
            <p style="margin: 0 0 8px 0; color: #667eea; font-weight: bold; font-size: 1.2em;">
                ${product['price']}
            </p>
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.95em;">
                <strong>Category:</strong> {product['category']}
            </p>
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.95em;">
                <strong>Description:</strong> {product['description']}
            </p>
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.9em;">
                <strong>Features:</strong> {', '.join(product['features'])}
            </p>
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.9em;">
                <strong>Colors:</strong> {', '.join(product['colors'])}
            </p>
            <p style="margin: 0 0 0 0; color: #666; font-size: 0.9em;">
                <strong>Sizes:</strong> {', '.join(product['sizes'])}
            </p>
        </div>
    </div>
</div>
"""
    
    return response


@tool
def get_all_categories() -> str:
    """
    Get a list of all available product categories.
    
    Returns:
        List of categories
    """
    categories = list(set(p['category'] for p in PRODUCT_CATALOG))
    return "Available categories:\n" + "\n".join(f"- {cat}" for cat in sorted(categories))


@tool
def filter_products_by_price(min_price: float, max_price: float) -> str:
    """
    Search for products online within a specific price range using Perplexity.

        Args:
        min_price: Minimum price
        max_price: Maximum price
    
    Returns:
        List of products in the price range from online stores
    """
    # Use Perplexity to search for products in price range
    query = f"products between ${min_price} and ${max_price} shopping online"
    filtered = perplexity_product_search(query, search_type="shopping")
    
    if not filtered:
        return f"No products found in the price range ${min_price} - ${max_price}."
    
    response = f"<div style='margin-bottom: 15px; color: #666;'>🔍 Found {len(filtered)} product(s) in the range ${min_price} - ${max_price} from online stores:</div>\n\n"
    
    # Add note about real-time results
    response += "<div style='background: #e8f4fd; border-radius: 8px; padding: 10px; margin-bottom: 15px; font-size: 0.9em; color: #1976d2;'>"
    response += "These are real products from online retailers in your price range. Click the links to view on the actual store."
    response += "</div>\n\n"
    
    for product in filtered:
        # Create HTML product card with image and link
        price_display = f"${product['price']}" if isinstance(product['price'], (int, float)) else product['price']
        
        product_html = f"""
<div style="background: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="display: flex; gap: 15px;">
        <div style="flex-shrink: 0;">
            <img src="{product.get('image_url', 'https://via.placeholder.com/300?text=Image')}" 
                 loading="lazy" decoding="async"
                 style="width: 120px; height: 120px; object-fit: contain; background: #fff; border-radius: 8px; border: 2px solid #ddd;"
                 alt="{product['name']}" 
                 onerror="this.src='https://via.placeholder.com/100?text=No+Image'">
        </div>
        <div style="flex: 1;">
            <h3 style="margin: 0 0 8px 0; color: #333; font-size: 1.1em;">
                <a href="{product.get('link_url', '#')}" target="_blank" style="color: #667eea; text-decoration: none;">
                    {product['name']} 🔗
                </a>
            </h3>
            <p style="margin: 0 0 8px 0; color: #667eea; font-weight: bold; font-size: 1.3em;">
                {price_display}
            </p>
            
            <p style="margin: 0 0 8px 0; color: #666; font-size: 0.95em;">
                <strong>Description:</strong> {product['description']}
            </p>
            <div style="margin-top: 10px;">
                <a href="{product.get('link_url', '#')}" target="_blank" 
                   style="background: #667eea; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; display: inline-block; font-size: 0.9em;">
                    View on Store →
                </a>
            </div>
        </div>
    </div>
</div>
"""
        response += product_html
    
    return response


# System prompt for the agent
AGENT_SYSTEM_PROMPT = """You are Rufus, an intelligent shopping assistant that searches REAL products from online stores.

Your capabilities:
1.  General Conversation : Answer questions about yourself, what you can do, and engage in friendly conversation
2.  Real-Time Product Search : Find actual products from online retailers with current prices and direct shopping links
3.  Image-Based Search : Analyze product images and find similar items available online
4.  Price Range Search : Find products within specific budget ranges from real stores

CRITICAL INSTRUCTIONS:
- You search REAL PRODUCTS from ACTUAL ONLINE STORES using Perplexity Search API
- You MUST use the search_products_by_text tool when users ask for products, recommendations, or searches
- You MUST use the filter_products_by_price tool when users mention price ranges
- NEVER describe products without using the tools first - the tools return HTML cards with images and shopping links
- The tools return FORMATTED HTML with real product images, current prices, and direct store links
- DO NOT REFORMAT OR SUMMARIZE THE TOOL OUTPUT - return it EXACTLY as provided
- When users ask for photos or links, use the search tools - they include real product images and store links
- Present the EXACT tool response to users without modification

Guidelines:
- Be friendly, helpful, and conversational
- ALWAYS use tools for product queries - you search real online stores
- Explain that you find REAL products from actual retailers, not a local catalog
- For product recommendations, you can ask clarifying questions, then use search_products_by_text
- The tool responses are COMPLETE with clickable links to buy - just return them AS-IS
- If you don't find exact matches, search for similar alternatives
- You can add a brief intro like "I found these products online:" but then show the EXACT tool output
- Remind users these are real products they can buy by clicking the links

Important: You search LIVE products from real online stores with current availability and pricing!

Remember: ALWAYS use the tools - they return real products with images and direct shopping links!"""


# Create the AI Agent
def create_commerce_agent():
    """Create the single AI agent that handles all use cases."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )
    
    tools = [
        search_products_by_text,
        get_product_by_id,
        get_all_categories,
        filter_products_by_price
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


# Initialize agent
print("Initializing AI agent...")
commerce_agent = create_commerce_agent()
print("AI agent ready!")


# Image-based search functionality
def search_products_by_image(image_data: str) -> Dict[str, Any]:
    """
    Search for products based on an image using multimodal LLM.

        Args:
        image_data: Base64 encoded image data
    
    Returns:
        Dictionary with search results
    """
    try:
        # Ensure an event loop exists in this Flask worker thread (fixes
        # "There is no current event loop in thread 'process_request_thread'")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Prefer BLIP for captioning. If unavailable, fall back to Gemini.
        raw_bytes = base64.b64decode(image_data)
        image_description = None
        if BLIP_AVAILABLE and load_blip_if_needed():
            image_description = caption_image_bytes(raw_bytes)
        if not image_description:
            # Fallback to Gemini vision if BLIP not available
            vision_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=GEMINI_API_KEY
            )
            prompt = "Describe this product briefly for shopping search (brand/model if visible, color, type, style)."
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
                ]
            )
            response = vision_llm.invoke([message])
            image_description = response.content
        
        # Convert verbose analysis to a clean product query (e.g., "black DC low-top skate shoe")
        query_from_image = extract_product_query_from_image_description(image_description)
        if not query_from_image:
            query_from_image = image_description
        # Use Perplexity for real-time product search based on the extracted query
        matched_products = perplexity_product_search(query_from_image, search_type="shopping")
        
        return {
            "success": True,
            "image_analysis": image_description,
            "query": query_from_image,
            "products": matched_products,
            "count": len(matched_products),
            "source": "Online search via Perplexity"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Flask Web Application
app = Flask(__name__)

# Enable CORS if available
if CORS_AVAILABLE:
    CORS(app)

# Store conversation history per session
conversation_histories = {}


# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Commerce AI Agent - Rufus</title>
    <style>
        :root {
            --bg: #0f172a;
            --bg-grad-1: #5b7cfa;
            --bg-grad-2: #7b4ea2;
            --panel: #0b1220;
            --panel-contrast: #ffffff;
            --muted: #94a3b8;
            --primary: #667eea;
            --primary-contrast: #ffffff;
            --border: #1f2a44;
        }
        body.light {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-contrast: #0f172a;
            --muted: #5b6b86;
            --border: #e6eaf2;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: radial-gradient(1200px 600px at 10% 10%, rgba(102,126,234,0.25), transparent 60%),
                        radial-gradient(900px 500px at 90% -10%, rgba(118,75,162,0.25), transparent 60%),
                        var(--bg);
            min-height: 100vh;
            display: flex; justify-content: center; align-items: center;
            padding: 24px;
            transition: background 300ms ease;
        }
        .container {
            background: var(--panel);
            color: var(--panel-contrast);
            border-radius: 18px;
            box-shadow: 0 30px 80px rgba(0,0,0,0.35), 0 8px 24px rgba(0,0,0,0.25);
            width: 100%; max-width: 900px; overflow: hidden;
            border: 1px solid var(--border);
        }
        .header {
            position: relative;
            background: linear-gradient(135deg, var(--bg-grad-1), var(--bg-grad-2));
            color: #fff; padding: 28px 28px 24px 28px;
        }
        .header h1 { font-size: 1.75em; letter-spacing: 0.5px; }
        .header p { opacity: 0.9; margin-top: 6px; }
        .header-actions { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
        .chip-btn { padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.6); background: rgba(255,255,255,0.08); color: #fff; font-size: 0.9em; cursor: pointer; transition: all 0.2s ease; }
        .chip-btn.active { background: #fff; color: #333; }
        .chip-btn:hover { background: rgba(255,255,255,0.15); }
        .tabs { display: flex; background: transparent; border-bottom: 1px solid var(--border); }
        .tab { flex: 1; padding: 14px; text-align: center; cursor: pointer; background: transparent; border: none; color: var(--muted); font-weight: 600; letter-spacing: 0.2px; }
        .tab:hover { color: var(--panel-contrast); background: rgba(102,126,234,0.06); }
        .tab.active { color: var(--panel-contrast); border-bottom: 3px solid var(--primary); background: rgba(102,126,234,0.08); }
        .tab-content { display: none; padding: 22px; }
        .tab-content.active { display: block; }
        .chat-container { height: 440px; overflow-y: auto; border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 18px; background: rgba(255,255,255,0.02); }
        .message { margin-bottom: 16px; padding: 14px 16px; border-radius: 16px; max-width: 85%; line-height: 1.6; font-size: 0.98em; word-wrap: break-word; }
        .message.user { background: var(--primary); color: #fff; margin-left: auto; text-align: right; box-shadow: 0 8px 18px rgba(102,126,234,0.35); }
        .message.agent { background: rgba(255,255,255,0.06); color: var(--panel-contrast); border: 1px solid var(--border); }
        .input-container { display: flex; gap: 12px; background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
        input[type="text"], textarea { flex: 1; padding: 14px 16px; border: 1px solid var(--border); background: transparent; color: var(--panel-contrast); border-radius: 10px; font-size: 0.98em; }
        input[type="text"]:focus, textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(102,126,234,0.18); }
        button { padding: 11px 16px; background: var(--primary); color: var(--primary-contrast); border: none; border-radius: 10px; font-size: 0.96em; cursor: pointer; transition: transform 120ms ease, background 200ms ease; }
        button:hover { background: #5b73e2; transform: translateY(-1px); }
        button:disabled { background: #9299b5; cursor: not-allowed; }
        .voice-toggle, .theme-toggle { padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.6); background: rgba(255,255,255,0.08); color: #fff; font-size: 0.9em; }
        .voice-toggle.active { background: #fff; color: #333; }
        #mic-button { padding: 10px 12px; min-width: auto; }
        #mic-button.listening { background: #e74c3c; animation: pulse 1.5s infinite; will-change: opacity; transform: translateZ(0); }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }
        .image-upload { border: 2px dashed var(--primary); border-radius: 12px; padding: 48px 24px; text-align: center; cursor: pointer; transition: all 0.3s ease; margin-bottom: 24px; background: rgba(102,126,234,0.05); position: relative; overflow: hidden; }
        .image-upload:hover { background: rgba(102,126,234,0.12); border-color: var(--primary); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.15); }
        .image-upload::before { content: '📷'; font-size: 2.5em; display: block; margin-bottom: 12px; opacity: 0.7; }
        .image-upload p { margin: 0; font-size: 1.1em; color: var(--panel-contrast); }
        .image-upload p:last-child { margin-top: 8px; font-size: 0.9em; color: var(--muted); }
        .image-preview { max-width: 100%; max-height: 320px; margin: 18px 0; border-radius: 12px; }
        .results { margin-top: 18px; }
        .product-card { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 18px; margin-bottom: 14px; border: 1px solid var(--border); transition: transform 140ms ease, box-shadow 200ms ease; }
        .product-card:hover { transform: translateY(-2px); box-shadow: 0 16px 36px rgba(0,0,0,0.35); }
        .product-card h3 { color: var(--panel-contrast); margin-bottom: 8px; }
        .product-card p { color: var(--muted); line-height: 1.6; }
        .price { font-size: 1.2em; color: var(--primary); font-weight: 800; margin: 6px 0 10px 0; }
        .loading { text-align: left; color: var(--primary); padding: 14px; }
        .typing { display: inline-flex; align-items: center; gap: 6px; }
        .typing .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--primary); opacity: 0.5; animation: typing-bounce 1s ease-in-out infinite; will-change: transform, opacity; transform: translateZ(0); }
        .typing .dot:nth-child(2) { animation-delay: .15s; }
        .typing .dot:nth-child(3) { animation-delay: .3s; }
        @keyframes typing-bounce { 0%, 100% { transform: translateY(0); opacity: .4; } 50% { transform: translateY(-5px); opacity: 1; } }
        input[type="file"] { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Commerce AI Agent</h1>
            <p>Your intelligent shopping assistant</p>
            <div class="header-actions">
                <button id="theme-toggle" class="chip-btn" onclick="toggleTheme()" title="Toggle theme">
                    🌙 Theme
                </button>
                <button id="voice-toggle" class="voice-toggle" onclick="toggleVoice()" title="Toggle voice responses">
                    🔊 Voice: <span id="voice-status">OFF</span>
                </button>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('chat')"> Chat</button>
            <button class="tab" onclick="switchTab('image')">Image Search</button>
        </div>
        
        <div id="chat-tab" class="tab-content active">
            <div class="chat-container" id="chat-messages">
                <div class="message agent">
                    Hi! I'm Palona, your shopping assistant. I can help you find products, answer questions, and make recommendations. What are you looking for today?
                </div>
            </div>
            <div class="input-container">
                <input type="text" id="user-input" placeholder="Ask me anything... Please ues: If you want to buy something please type I am looking for" onkeypress="handleKeyPress(event)">
                <button id="mic-button" onclick="toggleVoiceInput()" title="Voice input">🎤</button>
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div id="image-tab" class="tab-content">
            <div class="image-upload" onclick="document.getElementById('image-input').click()">
                <p>Click to upload an image or drag and drop</p>
                <p>Upload a product image to find similar items</p>
            </div>
            <input type="file" id="image-input" accept="image/*" onchange="handleImageUpload(event)">
            <div id="image-preview"></div>
            <div id="image-results"></div>
        </div>
    </div>
    
    <script>
        let sessionId = Math.random().toString(36).substring(7);
        let voiceEnabled = false;
        let recognition = null;
        let isListening = false;
        let isDarkMode = true; // Default to dark mode
        
        // Initialize theme from localStorage
        function initializeTheme() {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {
                isDarkMode = false;
                document.body.classList.add('light');
                document.getElementById('theme-toggle').innerHTML = '☀️ Theme';
            }
        }
        
        // Initialize on page load
        initializeTheme();
        
        // Initialize speech recognition if available
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById('user-input').value = transcript;
                document.getElementById('mic-button').classList.remove('listening');
                isListening = false;
                // Automatically send the message
                sendMessage();
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                document.getElementById('mic-button').classList.remove('listening');
                isListening = false;
            };
            
            recognition.onend = function() {
                document.getElementById('mic-button').classList.remove('listening');
                isListening = false;
            };
        }
        
        function toggleTheme() {
            isDarkMode = !isDarkMode;
            const body = document.body;
            const themeBtn = document.getElementById('theme-toggle');
            
            if (isDarkMode) {
                body.classList.remove('light');
                themeBtn.innerHTML = '🌙 Theme';
            } else {
                body.classList.add('light');
                themeBtn.innerHTML = '☀️ Theme';
            }
            
            // Save preference
            localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
        }
        
        function toggleVoice() {
            voiceEnabled = !voiceEnabled;
            const toggleBtn = document.getElementById('voice-toggle');
            const statusSpan = document.getElementById('voice-status');
            
            if (voiceEnabled) {
                toggleBtn.classList.add('active');
                statusSpan.textContent = 'ON';
            } else {
                toggleBtn.classList.remove('active');
                statusSpan.textContent = 'OFF';
            }
        }
        
        function toggleVoiceInput() {
            if (!recognition) {
                alert('Speech recognition is not supported in your browser. Please try Chrome or Edge.');
                return;
            }
            
            if (isListening) {
                recognition.stop();
                document.getElementById('mic-button').classList.remove('listening');
                isListening = false;
            } else {
                recognition.start();
                document.getElementById('mic-button').classList.add('listening');
                isListening = true;
            }
        }
        
        function speakText(text) {
            if (!voiceEnabled) return;
            
            // Extract only headings (e.g., product titles inside <h3>)
            let headingOnly = '';
            try {
                if (typeof text === 'string' && (text.includes('<h3') || text.includes('<div') || text.includes('<p'))) {
                    const temp = document.createElement('div');
                    temp.innerHTML = text;
                    const headings = Array.from(temp.querySelectorAll('h3'))
                        .map(h => (h.textContent || '').trim())
                        .filter(Boolean)
                        .slice(0, 5); // speak up to first 5 headings
                    if (headings.length > 0) {
                        headingOnly = headings.join(', ');
                    }
                }
            } catch (e) {
                // If parsing fails, do not speak
                headingOnly = '';
            }
            
            // If we didn't find headings, do not speak anything
            if (!headingOnly) {
                window.speechSynthesis.cancel();
                return;
            }
            
            // Cancel any ongoing speech and speak headings only
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(headingOnly);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        }
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function sendMessage() {
            console.log('sendMessage called');
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            console.log('Message:', message);
            
            if (!message) {
                console.log('Empty message, returning');
                return;
            }
            
            // Add user message to chat
            addMessageToChat('user', message);
            input.value = '';
            
            // Show loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message agent loading';
            loadingDiv.id = 'loading';
            loadingDiv.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
            document.getElementById('chat-messages').appendChild(loadingDiv);
            scrollToBottom();
            
            try {
                console.log('Sending fetch request...');
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Response data:', data);
                
                // Remove loading
                const loadingElement = document.getElementById('loading');
                if (loadingElement) {
                    loadingElement.remove();
                }
                
                // Add agent response
                if (data.success && data.response) {
                    addMessageToChat('agent', data.response);
                    // Speak the response if voice is enabled
                    speakText(data.response);
                } else {
                    console.error('Invalid response format:', data);
                    const errorMsg = 'Sorry, I got an invalid response. Please try again.';
                    addMessageToChat('agent', errorMsg);
                    speakText(errorMsg);
                }
                
            } catch (error) {
                console.error('Chat error:', error);
                const loadingElement = document.getElementById('loading');
                if (loadingElement) {
                    loadingElement.remove();
                }
                const errorMsg = 'Sorry, I encountered an error. Please try again.';
                addMessageToChat('agent', errorMsg);
                speakText(errorMsg);
            }
        }
        
        function addMessageToChat(sender, message) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            // Check if message contains HTML (product cards or other rich content)
            if (message.includes('<div') || message.includes('<p') || message.includes('<img') || message.includes('```html')) {
                // If it has ```html wrapper, extract the content
                if (message.includes('```html')) {
                    message = message.replace(/```html\\n?/g, '').replace(/```\\n?/g, '');
                }
                // Render HTML directly for rich content
                messageDiv.innerHTML = message;
            } else {
                // Plain text - replace newlines with br tags
                messageDiv.innerHTML = message.replace(/\\n/g, '<br>');
            }
            
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function scrollToBottom() {
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        async function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('image-preview').innerHTML = 
                    `<img src="${e.target.result}" class="image-preview" alt="Uploaded image">`;
            };
            reader.readAsDataURL(file);
            
            // Show loading with animated typing dots
            document.getElementById('image-results').innerHTML = 
                '<div class="loading">' +
                '<div style="margin-bottom:6px; color: var(--muted);">This is an image search, it may take a moment — please wait and see the magic happen</div>' +
                '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>' +
                '</div>';
            
            // Send to backend
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('/search-by-image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayImageSearchResults(data);
                } else {
                    document.getElementById('image-results').innerHTML = 
                        `<div class="message agent">Error: ${data.error}</div>`;
                }
                
            } catch (error) {
                document.getElementById('image-results').innerHTML = 
                    '<div class="message agent">Sorry, I encountered an error processing the image.</div>';
            }
        }
        
        function displayImageSearchResults(data) {
            let html = '<div class="results">';
            html += `<h3>Image Analysis:</h3>`;
            html += `<p style="margin-bottom: 20px; color: #666;">${data.image_analysis}</p>`;
            
            // Add Perplexity market insights if available
            if (data.market_insights && Object.keys(data.market_insights).length > 0) {
                const insights = data.market_insights;
                if (insights.price_trends && insights.price_trends.length > 0) {
                    html += '<div style="background: #f0f4f8; border-radius: 10px; padding: 15px; margin: 15px 0;">';
                    html += '<h4 style="color: #667eea; margin: 0 0 10px 0;">💰 Current Market Prices:</h4>';
                    html += `<p style="margin: 0; color: #666;">${insights.price_trends[0].snippet}</p>`;
                    html += `<a href="${insights.price_trends[0].url}" target="_blank" style="color: #667eea; font-size: 0.9em;">Learn more →</a>`;
                    html += '</div>';
                }
            }
            
            html += `<h3>Found ${data.count} matching product(s) online:</h3>`;
            
            // Add note about real-time results
            html += '<div style="background: #e8f4fd; border-radius: 8px; padding: 10px; margin: 10px 0; font-size: 0.9em; color: #1976d2;">';
            html += 'These are real products from online retailers based on your image. Click to view on the actual store.';
            html += '</div>';
            
            data.products.forEach(product => {
                const priceDisplay = typeof product.price === 'number' ? `$${product.price}` : product.price;
                
                html += `
                    <div class="product-card" style="display: flex; gap: 15px; background: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="flex-shrink: 0;">
                            <img src="${product.image_url || 'https://via.placeholder.com/300?text=Image'}" 
                                 loading="lazy" decoding="async"
                                 style="width: 120px; height: 120px; object-fit: contain; background: #fff; border-radius: 8px; border: 2px solid #ddd;"
                                 alt="${product.name}"
                                 onerror="this.src='https://via.placeholder.com/100?text=No+Image'">
                        </div>
                        <div style="flex: 1;">
                            <h3 style="margin: 0 0 8px 0;">
                                <a href="${product.link_url || '#'}" target="_blank" style="color: #667eea; text-decoration: none;">
                                    ${product.name} 
                                </a>
                            </h3>
                            <div class="price" style="color: #667eea; font-weight: bold; font-size: 1.3em; margin-bottom: 8px;">${priceDisplay}</div>
                            <p style="margin: 5px 0; color: #666;"><strong>Description:</strong> ${product.description}</p>
                            <p style="margin: 5px 0; color: #666;"><strong>Features:</strong> ${product.features ? product.features.join(', ') : 'Check product page'}</p>
                            <div style="margin-top: 10px;">
                                <a href="${product.link_url || '#'}" target="_blank" 
                                   style="background: #667eea; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; display: inline-block; font-size: 0.9em;">
                                    View on Store →
                                </a>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            document.getElementById('image-results').innerHTML = html;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the main web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')

        # Get or create conversation history
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []

        chat_history = conversation_histories[session_id]

        # Intent detection first: general chat vs product/knowledge
        if is_greeting_or_chitchat(user_message) or (not is_product_intent(user_message) and not is_knowledge_question(user_message)):
            try:
                response_text = get_smalltalk_response(user_message, chat_history)
                # Update history
                chat_history.append(HumanMessage(content=user_message))
                chat_history.append(AIMessage(content=response_text))
                # Trim history
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
                conversation_histories[session_id] = chat_history
                return jsonify({
                    "success": True,
                    "response": response_text,
                    "conversation": True
                })
            except Exception as e:
                print(f"Smalltalk path error: {e}")
                # Fall-through to agent or search below

        # Product or knowledge intent: try agent first
        try:
            if USE_GEMINI_AGENT and time.time() > GEMINI_BLOCK_UNTIL:
                response = commerce_agent.invoke({
                    "input": user_message,
                    "chat_history": chat_history
                })
                # Update history
                chat_history.append(HumanMessage(content=user_message))
                chat_history.append(AIMessage(content=response['output']))
                # Trim history
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]
                conversation_histories[session_id] = chat_history
                return jsonify({
                    "success": True,
                    "response": response['output']
                })
            else:
                if is_product_intent(user_message):
                    fallback_html = search_products_by_text(user_message)
                    return jsonify({
                        "success": True,
                        "response": fallback_html,
                        "fallback": True
                    })
                else:
                    response_text = get_smalltalk_response(user_message, chat_history)
                    return jsonify({
                        "success": True,
                        "response": response_text,
                        "conversation": True
                    })
        except Exception as agent_error:
            try:
                if is_product_intent(user_message):
                    fallback_html = search_products_by_text(user_message)
                    return jsonify({
                        "success": True,
                        "response": fallback_html,
                        "fallback": True,
                        "reason": str(agent_error)[:200]
                    })
                else:
                    response_text = get_smalltalk_response(user_message, chat_history)
                    return jsonify({
                        "success": True,
                        "response": response_text,
                        "conversation": True,
                        "fallback": True
                    })
            except Exception as fallback_error:
                return jsonify({
                    "success": False,
                    "error": f"{agent_error} | fallback_error: {fallback_error}"
                }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/search-by-image', methods=['POST'])
def search_by_image():
    """Handle image-based product search."""
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_file = request.files['image']
        
        # Read and encode image
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Search products
        results = search_products_by_image(image_data)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/products', methods=['GET'])
def get_products():
    """API endpoint to get all products."""
    return jsonify({
        "success": True,
        "products": PRODUCT_CATALOG,
        "count": len(PRODUCT_CATALOG)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "agent": "ready",
        "products_loaded": len(PRODUCT_CATALOG)
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Commerce AI Agent - Ready!")
    print("="*50)
    print(f"\nLoaded {len(PRODUCT_CATALOG)} products")
    print("Agent capabilities:")
    print(" ✓ General conversation")
    print(" ✓ Text-based product recommendations")
    print(" ✓ Image-based product search")
    print("\nStarting web server...")
    print("   Local: http://localhost:3000")
    print("\n" + "="*50 + "\n")
    
    app.run(host='0.0.0.0', port=3000, debug=False, use_reloader=False)
