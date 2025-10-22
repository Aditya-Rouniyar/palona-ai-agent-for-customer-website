# Commerce AI Agent - API Documentation

Welcome to our Commerce AI Agent API! This guide will walk you through everything you need to know to integrate with our smart shopping assistant. Whether you're building a mobile app, adding features to your website, or just curious about how it works, you'll find everything you need right here.

Base URL: `http://localhost:3000`

# What Can This AI Do?

Our agent is pretty smart! It can:
- Chat naturally - Have real conversations, answer questions, and be helpful
- Find products - Search for items using text descriptions like "gaming laptop under $1000"
- Image search - Upload a photo and find similar products
- Stay healthy - Let you check if everything is working properly

# Let's Talk About the Endpoints

# 1. Chat with the AI 

 POST  `/chat`

This is where the magic happens! Send a message to our AI and get a smart response back.

# How to Use It

 Headers: 
```
Content-Type: application/json
```

 Request Body: 
```json
{
  "message": "Hi, can you recommend a good pair of running shoes?",
  "session_id": "user123abc"
}
```

 What you need to send: 
- `message` (required): Whatever you want to say to the AI
- `session_id` (required): A unique ID to keep track of your conversation

# What You'll Get Back

 Success Response: 
```json
{
  "success": true,
  "response": "Hello! I'd be happy to help you find running shoes. What's your budget and what type of running do you do?",
  "conversation": true,
  "fallback": false
}
```

 When the AI finds products: 
```json
{
  "success": true,
  "response": "<div style='margin-bottom: 15px; color: #666;'> Found 5 product(s) from online stores:</div>\n\n<div style='background: #e8f4fd; border-radius: 8px; padding: 10px; margin-bottom: 15px; font-size: 0.9em; color: #1976d2;'>These are real products from online retailers with current pricing and availability. Click the links to view on the actual store.</div>\n\n<div style=\"background: #f9f9f9; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);\">...</div>",
  "fallback": false
}
```

 If something goes wrong: 
```json
{
  "success": false,
  "error": "Oops! Something went wrong. Please try again."
}
```

# Response Fields Explained

- `success`: `true` if everything worked, `false` if there was a problem
- `response`: The AI's reply (could be text or HTML with product cards)
- `conversation`: `true` for general chat, `false` for product recommendations
- `fallback`: `true` if we had to use a backup method
- `reason`: Why we used a fallback (only when `fallback` is `true`)
- `error`: What went wrong (only when `success` is `false`)

# 2. Search Products by Image 

 POST  `/search-by-image`

Upload a photo and find similar products! This is super cool - just snap a picture of something you like and we'll find similar items for you.

# How to Use It

 Headers: 
```
Content-Type: multipart/form-data
```

 Request Body: 
```
image: [your image file]
```

 Example with curl: 
```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:3000/search-by-image
```

# What You'll Get Back

 Success Response: 
```json
{
  "success": true,
  "image_analysis": "A black low-top skate shoe with white laces and sole.",
  "query": "black low-top skate shoe",
  "products": [
    {
      "id": "perplexity_0",
      "name": "Vans Old Skool Black/White",
      "price": 65.0,
      "category": "Online Product",
      "description": "Classic skate shoe with durable canvas and suede upper...",
      "image_url": "https://example.com/vans.jpg",
      "link_url": "https://www.vans.com/oldskool"
    }
  ],
  "count": 1,
  "source": "Online search via Perplexity"
}
```

 If no image is provided: 
```json
{
  "success": false,
  "error": "No image provided"
}
```

# Response Fields Explained

- `success`: Did it work?
- `image_analysis`: What the AI thinks it sees in your image
- `query`: The search terms we used to find products
- `products`: Array of similar products we found
- `count`: How many products we found
- `source`: Where we got the results from

# 3. Check if Everything is Working 

 GET  `/api/health`

Quick health check to see if our service is running properly.

# How to Use It

 Example: 
```bash
curl http://localhost:3000/api/health
```

# What You'll Get Back

```json
{
  "status": "healthy",
  "agent": "ready",
  "products_loaded": 15
}
```

 What this means: 
- `status`: "healthy" means everything is working great
- `agent`: "ready" means the AI is ready to chat
- `products_loaded`: How many products we have in our local catalog

# 4. Get All Products (For Developers) 

 GET  `/api/products`

This endpoint returns our entire product catalog. It's mainly for debugging or if you want to see what products we have locally.

# How to Use It

```bash
curl http://localhost:3000/api/products
```

# What You'll Get Back

```json
{
  "success": true,
  "products": [
    {
      "id": "1",
      "name": "Wireless Bluetooth Earbuds",
      "price": 49.99,
      "category": "Electronics",
      "description": "High-quality sound with comfortable fit...",
      "image_url": "https://images.unsplash.com/photo-1607863680198-6e9191a70f0f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NzEyNjZ8MHwxfHNlYXJjaHwxfHxlYXJidWRzfGVufDB8MHx8fDE3MTY5OTY2NDd8MA&ixlib=rb-4.0.3&q=80&w=1080",
      "link_url": "https://www.amazon.com/earbuds"
    }
  ],
  "count": 15
}
```

# Real-World Examples

# Example 1: General Conversation
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, what can you do?", "session_id": "user123"}'
```

 Response: 
```json
{
  "success": true,
  "response": "Hello! I'm Palona, your shopping assistant. I can help you find products, answer questions, and have conversations. What would you like to do today?",
  "conversation": true
}
```

# Example 2: Product Search
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me a gaming laptop under $1000", "session_id": "user123"}'
```

 Response: 
```json
{
  "success": true,
  "response": "<div style='margin-bottom: 15px; color: #666;'> Found 3 product(s) from online stores:</div>...",
  "conversation": false,
  "fallback": false
}
```

# Example 3: Image Search
```bash
curl -X POST http://localhost:3000/search-by-image \
  -F "image=@laptop.jpg"
```

 Response: 
```json
{
  "success": true,
  "image_analysis": "A sleek silver laptop with a black keyboard",
  "query": "silver laptop",
  "products": [...],
  "count": 5
}
```

# Error Handling

We try to be helpful even when things go wrong! Here are some common scenarios:

# API Quota Exceeded
If we hit our API limits, we'll automatically fall back to our local product catalog:
```json
{
  "success": true,
  "response": "...",
  "fallback": true,
  "reason": "LLM agent failed due to quota"
}
```

# Invalid Input
If you send something we can't understand:
```json
{
  "success": false,
  "error": "Invalid request format"
}
```

# Service Unavailable
If one of our AI services is down:
```json
{
  "success": false,
  "error": "AI service temporarily unavailable"
}
```

# Rate Limits & Best Practices

# Rate Limits
-  Chat endpoint : No strict limits, but be reasonable
-  Image search : No strict limits, but processing takes time
-  Health check : Feel free to check as often as you want

# Best Practices
1.  Use session IDs : Keep track of conversations for better context
2.  Handle errors gracefully : Always check the `success` field
3.  Be patient with image search : It takes a few seconds to process
4.  Cache responses : Product results don't change that often

# Integration Examples

# JavaScript (Frontend)
```javascript
// Chat with the AI
async function chatWithAI(message, sessionId) {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      session_id: sessionId
    })
  });
  
  const data = await response.json();
  return data;
}

// Search by image
async function searchByImage(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('/search-by-image', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  return data;
}
```

# Python (Backend)
```python
import requests

# Chat with the AI
def chat_with_ai(message, session_id):
    response = requests.post('http://localhost:3000/chat', json={
        'message': message,
        'session_id': session_id
    })
    return response.json()

# Search by image
def search_by_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:3000/search-by-image', files=files)
    return response.json()
```

# Troubleshooting

# Common Issues

 "Connection refused" 
- Make sure the server is running on port 3000
- Check if another service is using that port

 "Invalid JSON" 
- Make sure you're sending proper JSON in the request body
- Check that Content-Type is set to application/json

 "No image provided" 
- Make sure you're sending the image file in the correct format
- Use multipart/form-data for image uploads

 "AI service unavailable" 
- Check your API keys are set correctly
- The service might be temporarily down

# Getting Help

If you're still having issues:
1. Check the health endpoint first: `GET /api/health`
2. Look at the error messages - they're usually helpful
3. Make sure your request format matches the examples
4. Try a simple request first to test connectivity


 The API is designed to be simple and intuitive, so you should be able to get started quickly. 
