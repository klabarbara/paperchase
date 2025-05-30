"""
eval set for paperchase
the following are dummies pending real data

structure:
    EVAL_SET = [
     {
            "query": "string",
            "gold_ids": ["2404.01234", "2311.05678"],  # arXiv IDs (no 'cs.' prefix)
            "answer": "short natural-language answer the paper(s) should support"
        },
        ...
    ]
"""
EVAL_SET = [
    {
        "query": (
            "What computer-science papers explain how to build a Mandarin-English "
            "retrieval-augmented generation (RAG) pipeline?"
        ),
        "gold_ids": ["2403.12345", "2308.54321"],
        "answer": (
            "They should describe bilingual or cross-lingual retrieval techniques, "
            "e.g. using shared multilingual embeddings or alignment objectives."
        ),
    },
    # 15-50 more examples give you stable metrics
]