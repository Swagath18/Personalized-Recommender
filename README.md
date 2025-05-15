# LLM-Powered Personalized Product Recommender

A context-aware recommendation system powered by **Retrieval-Augmented Generation (RAG)** and **GPT-4**, enhanced with **hybrid reranking** and a **feedback sentiment analysis system** using VADER.

##  Features

- Semantic product search with FAISS + Sentence Transformers
- GPT-4 for natural language product recommendations
- Personalized reranking using user context + cosine similarity
- Multi-tab UI with Gradio (Recommendations + Feedback)
- Feedback logging with real-time sentiment analysis (VADER)
- Logged query, prompt, and model responses for auditability

## Stack

- Python 
- SentenceTransformers (MiniLM-L6-v2)
- FAISS (vector search)
- OpenAI (GPT-4 API)
- VADER Sentiment Analysis
- Gradio (for UI)
- Pandas, Numpy, Scikit-learn

## Folder Structure

```
.
├── llmpoweredrag.py                 # Main app script
├── amazon_electronics_products.csv # Sample dataset
├── .env                             # API keys
├── recommendation_log.txt          # GPT prompt/response logs
├── feedback_log.txt                # Feedback + sentiment logs
└── README.md
```

## Setup Instructions

```bash
git clone https://github.com/yourusername/llm-personalized-recommender.git
cd llm-personalized-recommender
pip install -r requirements.txt
python llmragsenti.py
```

Make sure your `.env` file contains:

```
OPENAI_API_KEY=your-api-key-here
```

## Try These Test Cases

**Query:**  
`best noise-cancelling headphones for Zoom calls`

**Context:**  
```
prefers Bose or Sony
needs comfort for extended meetings
wants good microphone quality
```

Click "Recommend" and observe the GPT-4 output with ranked FAISS items.

Then test feedback like:
```
The suggestions made no sense. I didn’t find them helpful.
```

## Output Example

```text
Retrieved Products:
Sony WH-1000XM4 (Context Score: 0.874)
Bose QC35 II (Context Score: 0.832)

GPT-4 Recommendations:
1. Sony WH-1000XM4 - Superior noise cancellation...
2. Bose QC35 II - Excellent comfort and clarity...
```

## Sentiment Feedback Log

Each entry is saved with timestamp and label:
```
[2025-04-24 15:06:36] User Feedback: Loved it! | Sentiment: positive (0.89)
```

## License

MIT

##  Future Improvements

- Add image-based product cards
- Deploy on Hugging Face Spaces or Streamlit Cloud
- Build dashboard with feedback insights

