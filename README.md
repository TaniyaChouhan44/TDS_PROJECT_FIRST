# TDS VIRTUAL TA PROJECT

 # 🤖 TDS Virtual TA – Teaching Assistant Auto-Responder

**TDS Virtual TA** is a smart API-based assistant designed to automatically answer student questions in the **Tools in Data Science (TDS)** course offered by **IIT Madras B.S. in Data Science**.

It leverages course content and Discourse forum discussions to generate relevant, helpful responses — acting like a virtual teaching assistant for your peers.

---

## 📌 Project Purpose

This application was built to:
- Automatically answer student queries from the TDS Discourse forum (Jan–Apr 2025)
- Help lighten the load on real teaching assistants
- Learn about web scraping, LLM prompting, and API deployment

---

## 🚀 Deployment

- 🌐 **API Endpoint**: `https://tds-p1-production-8094.up.railway.app`
- ☁️ **Deployed via**: [Railway](https://railway.app/)
  - ✅ Switched to Railway after initial issues with Vercel’s timeout limits
- 🧪 Supports POST requests with JSON input containing:
  - `question`: string
  - `image`: optional base64 image (screenshot)

---

## 🛠️ How It Works

### 💡 Input (POST request):

```json
{
  "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
  "image": "<base64-encoded screenshot>"
}

### 💡 Output 
{
  "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
      "text": "Use the model that’s mentioned in the question."
    },
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
      "text": "You just have to use a tokenizer like Prof. Anand did, to get the number of tokens and multiply that by the given rate."
    }
  ]
}
