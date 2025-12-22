# Commerce AI Agent

### I have exposed the API keys because since it's a project and doesn't affect me.

### I have used Cursor as my coding partner to complete this project.


This is a smart shopping assistant that can chat with you, find products, and even understand images. Think of it as your personal shopping buddy that never gets tired of helping you find exactly what you're looking for.

### Note:- I have used code space to run the project live so it.

# What Can This Thing Do?
-  Chat naturally  - Have real conversations, ask questions, get helpful answers
-  Find products  - Tell it what you want and it'll search the web for real products with current prices
-  Image search  - Upload a photo of something you like and find similar items not only in the local catalog.
-  Voice chat  - Talk to it and it'll talk back (if you want it to)
-  Remember stuff  - Keeps track of your conversation so it makes sense

# User Experience
-  Looks good  - Clean, modern interface that works on your phone or computer
-  Fast results  - Gets product info from real stores in real-time
-  Rich product cards  - Shows you pictures, prices, and links to buy stuff
-  Smart routing  - Knows when you want to chat vs. when you want to shop
-  Handles errors gracefully  - Won't break if something goes wrong

# How It Works
This is built as one smart agent that handles everything. Instead of having separate bots for different tasks, it's all one system that figures out what you want and responds appropriately.

Here's how it's put together:
```

Web Frontend             Flask Server                    AI Services       
Chat Interface           Intent Detection                Gemini LLM    
Voice Input     -->      Session Mgmt           -->      Perplexity API
Image Upload             API Endpoints                   BLIP Vision   
Theme Toggle             Error Handling                  Local Catalog 

```

# Backend 
-  Python 3.8+  - The programming language everything is written in
-  Flask  - A lightweight web framework that handles all the web stuff
-  Flask-CORS  - Makes sure the frontend can talk to the backend

 Why Flask? 
- It's simple and doesn't get in your way
- Perfect for AI applications
- Easy to deploy and scale
- Great for rapid prototyping

# AI & Machine Learning (The Intelligence)
-  LangChain  - Framework that helps organize all the AI tools
-  Google Gemini 2.0 Flash  - The main AI that handles conversations and reasoning
-  Perplexity API  - Searches the web in real-time for live product data
-  BLIP  - Looks at images and describes what it sees
-  Transformers  - Library that makes BLIP work

 Why This AI Stack? 
-  Gemini 2.0 Flash  - Fast, cheap, and great at conversations
-  Perplexity  - ### Gets real, up-to-date product info from actual stores
-  BLIP  - ### Processes images locally (faster and more private)
-  LangChain  - Standard way to build AI agents

# Data & Storage (The Memory)
-  In-Memory Caching  - Stores recent results so it's faster next time
-  Session Management  - Remembers your conversation
-  JSON Catalog  - Backup product data in case the internet is down

 Why In-Memory? 
- Super fast access
- Simple to implement
- Perfect for development and small deployments

# Frontend (What You See)
-  HTML5/CSS3  - Modern web standards with fancy layouts
-  Vanilla JavaScript  - No heavy frameworks, just clean code
-  Web Speech API  - Built-in browser speech recognition and synthesis
-  CSS Variables  - Easy theme switching between light and dark modes

 Why Vanilla JavaScript? 
- No build process needed
- Works in any browser
- Loads fast
- Easy to understand and modify

# External Services (The Helpers)
-  Google Generative AI API  - Powers the conversation
-  Perplexity Search API  - Finds real products online
-  Unsplash  - Provides nice product images
-  HTTPX  - Makes efficient API calls

### What We Need
- Python 3.8 or higher
- pip (comes with Python)
- A modern web browser
- Internet connection (for the AI services)

# Step 1: Get the Code
```bash
git clone <repository-url>
cd palona
```

# Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

# Step 3: Set Up API Keys
You'll need API keys for the AI services. Set them as environment variables:

`GEMINI_API_KEY` 
`PERPLEXITY_API_KEY`
`USE_GEMINI_AGENT` = `true` 
`USE_LLM_FOR_CHITCHAT` = `true`
`USE_PERPLEXITY_FOR_CHITCHAT` = `true` 

# Step 4: Run It
```bash
python main.py
```

Open your browser and go to `http://localhost:3000` - you're ready to go!

# How to Use It

# Web Interface
1. Open `http://localhost:3000` in your browser
2. Use the  Chat  tab to talk to the AI
3. Use the  Image Search  tab to upload photos
4. Toggle the  Theme  button for light/dark mode
5. Enable  Voice  if you want to talk to it

# API Usage
Check out [API.md](API.md) for detailed API documentation.

# When Things Go Wrong
-  API quota exceeded  → Falls back to local product catalog
-  Service down  → Shows helpful error message
-  Bad input  → Asks you to try again
-  Network issues  → Uses cached results if available

# Making It Fast

# Caching Strategy
-  Search results : Cached for 5 minutes (products don't change that fast)
-  Image analysis : Cached so identical images don't need reprocessing
-  API responses : Smart caching to reduce external calls

# Speed Optimizations
-  Parallel processing : Makes multiple API calls at the same time
-  Immediate feedback : Shows "thinking..." while processing
-  Local processing : BLIP runs on your machine (faster than API calls)
-  Efficient queries : Optimized search terms for better results

## Testing It Out

# Manual Testing
1.  General conversation  - Try different greetings and questions
2.  Product search  - Test various product categories and price ranges
3.  Image search  - Upload different types of product images
4.  Voice interface  - Test speech recognition and synthesis
5.  Error handling  - Try invalid inputs and see what happens

# API Testing
```bash
# Test all endpoints
curl http://localhost:3000/api/health
curl -X POST http://localhost:3000/chat -H "Content-Type: application/json" -d '{"message":"test"}'
```


# Local Development
kill all the instances that were running
lsof -ti:3000 | xargs kill -9 2>/dev/null

```bash
python main.py
```

# Need Help?

If you run into issues:
1. Check the [API Documentation](API.md)
2. Look at the error logs in the console
3. Verify your API keys are set correctly
4. Test with the health check endpoint


