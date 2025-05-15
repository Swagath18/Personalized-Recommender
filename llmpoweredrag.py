

import os
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import gradio as gr
import datetime

# Load environment variables
load_dotenv()

# ========== Configuration ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "Missing OPENAI_API_KEY"

data = pd.read_csv("amazon_electronics_products.csv")

# ========== Embeddings ==========
model = SentenceTransformer("all-MiniLM-L6-v2")
descriptions = (data['title'] + ". " + data['description']).tolist()
embeddings = model.encode(descriptions, convert_to_tensor=False)

# Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ========== Core Recommendation Function with Reranking ==========
def generate_recommendations(query: str, user_context: List[str] = []) -> Tuple[str, str]:
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), k=5)
    retrieved_items = data.iloc[I[0]].copy()
    retrieved_embeddings = np.array([embeddings[i] for i in I[0]])

    if user_context:
        context_embedding = model.encode([" ".join(user_context)])[0]
        context_similarities = cosine_similarity([context_embedding], retrieved_embeddings)[0]
        retrieved_items["context_score"] = context_similarities
        top_items = retrieved_items.sort_values(by="context_score", ascending=False).head(2)
    else:
        top_items = retrieved_items.head(2)

    context = "\n".join(top_items['description'].tolist() + user_context)

    prompt = f"""
    Based on the following content:
    {context}

    Generate 2 personalized recommendations for a user interested in \"{query}\".
    Provide titles and brief descriptions.
    """

    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Log input and output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("recommendation_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}]\nQuery: {query}\nContext: {user_context}\nPrompt:\n{prompt}\nResponse:\n{response.choices[0].message.content.strip()}\n\n")

    # Display retrieved product titles + scores safely
    product_info = "\n".join([
        f"{row['title']} (Context Score: {float(row['context_score']):.3f})" if 'context_score' in row and isinstance(row['context_score'], (int, float, np.float64)) else f"{row['title']} (Context Score: N/A)"
        for _, row in top_items.iterrows()
    ])

    return product_info, response.choices[0].message.content.strip()

# ========== Gradio UI ==========
def gradio_interface(query, context):
    if not query.strip():
        return "", "Please enter a query to get recommendations."

    user_context = [line.strip() for line in context.split("\n") if line.strip()]
    product_info, recommendations = generate_recommendations(query, user_context)
    return product_info, recommendations

def feedback_handler(feedback: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User Feedback: {feedback}\n")
    return "Thanks for your feedback!"

main_ui = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="User Query"),
        gr.Textbox(label="User Context (optional, newline-separated)", lines=4)
    ],
    outputs=[gr.Textbox(label="Retrieved Products with Scores"), gr.Textbox(label="GPT-4 Recommendations")],
    title="LLM-Powered Personalized Recommender (Context-Aware + Feedback Enabled)",
    description="Click the Recommend button to get personalized product suggestions powered by vector search + GPT-4. Includes context-aware reranking, prompt logging, and feedback tracking.",
    live=False,  # disables real-time
    allow_flagging="never"
)

feedback_ui = gr.Interface(
    fn=feedback_handler,
    inputs=gr.Textbox(label="Your Feedback (What did you like or dislike?)"),
    outputs="text",
    title="Give Feedback",
    description="Help us improve the recommender by submitting your thoughts!"
)

demo = gr.TabbedInterface(
    [main_ui, feedback_ui],
    tab_names=["Product Recommender", "Feedback"]
)

if __name__ == "__main__":
    demo.launch()