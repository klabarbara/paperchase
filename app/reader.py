import hashlib, json, openai

class Reader:
    """
    given (query, doc) pair, produces:
    1) concise, 3 sentence summary
    2) "why this paper matters" explanation 

    responses cached to avoid excessive llm calls
    """
    def __init__(self, cache_path="cache.jsonl", model="gpt-4o"):
        self.model, self.cache = model, {}

    def __call__(self, query, doc):
        key = hashlib.sha1(f"{query}:{doc['id']}".encode()).hexdigest()
        
        if key in self.cache: return self.cache[key]

        prompt = f"""You are a research assistant ...
        Query: {query}
        Paper: {doc['title']} ({doc['year']})
        Abstract: {doc['abstract']}
        ---
        1. Give a concise 3‑sentence summary.
        2. Explain why this paper is helpful for the query; cite exact sections or line numbers."""

        resp = openai.ChatCompletion.create(model=self.model, messages=[{"role":"user","content":prompt}])
        self.cache[key] = resp.choices[0].message.content
    
        return self.cache[key]