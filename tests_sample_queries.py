"""Run sample queries against the running API and save results.

Start the API first:  uvicorn app:app --reload
Then run:             python tests_sample_queries.py
"""
import json
import os

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

QUERIES = [
    # From the assignment
    ("in-scope", "What is the core definition of Agentic AI as outlined in the eBook?"),
    ("in-scope", "What are the main architectural components required to build agentic systems?"),
    ("in-scope", "What real-world industry use cases for Agentic AI are discussed in the eBook?"),
    ("in-scope", "How does Agentic AI differ from traditional generative AI chatbots according to the text?"),
    ("in-scope", "What key challenges or limitations of Agentic AI are mentioned in the document?"),
    ("OUT-OF-SCOPE", "What is the capital of France?"),
    # From the reference guide
    ("in-scope", "What role does memory play in Agentic AI workflows?"),
    ("OUT-OF-SCOPE", "Who won the 2022 FIFA World Cup?"),
]


def main():
    results = []
    for kind, query in QUERIES:
        resp = requests.post(API_URL, json={"query": query}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        results.append(data)
        print(f"\n[{kind}] {query}")
        print(f"Answer    : {data['final_answer']}")
        print(f"Confidence: {data['confidence_score']}")
        print(f"Chunks    : {len(data['retrieved_context_chunks'])} retrieved")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nFull results saved to test_results.json")


if __name__ == "__main__":
    main()
