from __future__ import annotations

import traceback
from pathlib import Path
import gradio as gr

from src.chains.retrieval_chain import build_retrieval_chain
from src.chains.summary_chain import build_summary_chain

# initialize once for the whole session
retrieval_chain = build_retrieval_chain()
summary_chain   = build_summary_chain()

# turns Document into markdown
def _format_doc(doc, idx: int, add_summary: bool) -> str:
    header = f"### {idx + 1}. {doc.metadata.get('title', '(no title)')}\n"
    meta = f"Published: {doc.metadata.get('published', 'unknown')}\n\n"

    body = ""

    if add_summary:
        # map reduce summarizes over single doc
        summary = summary_chain.invoke({"input_documents": [doc]})["output_text"]
        body += f"**Summary**\n\n{summary}\n\n"
    else:
        # shows abstract excerpt to provide context if summary disabled (conserving tokens)
        excerpt = doc.page_content.strip()[:1000]
        body += f"{excerpt}...\n\n"

    return header + meta + body

# core search function calls RAG backend and formats rseult
def run_query(query: str, k: int, want_summary: bool) -> str:
    query = query.strip()
    if not query:
        return "Please enter a query first!"
    try:
        result = retrieval_chain.invoke(query)
        docs = result["docs"][: k]

        if not docs:
            return "No documents found! Try rephrasing your query."
        
        blocks = [_format_doc(d, idx, want_summary) for idx, d in enumerate(docs)]
        return "\n---\n".join(blocks) 
    
    except Exception as e:
        # exception exposed in UI for debugging 
        # TODO: replace with friendlier message
        tb = traceback.format_exc()
        return f"**ERROR** {e}\n\n```\n{tb}\n```"


# gradio interface

title_md = "# Paperchase - arXiv RAG Search"
subtitle_md = "## Let's get that read!"

# TODO: make legible
description_md = (
    "Enter a natural-language question about computer-science research. "
    "Paperchase will:\
        1. Extract arXiv search keywords via GPT-4o,\
        2. Grab candidate papers from arXiv,\
        3. Embed and rank them with Chroma,\
        4. (Optionally) summarise the top-K hits with GPT-4o,\
        5. If summary not chosen, abstract excerpt displayed instead.\
        NOTE: While in dev, k will be limited to a max of 3 due to\
        the constraints of the specific wrapper API in use. Will open\
        up once we go open source."
)

with gr.Blocks(title="Paperchase arXiv RAG Search") as demo:
    gr.Markdown(title_md)
    gr.Markdown(subtitle_md)
    gr.Markdown(description_md)

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="e.g. What are some ways to reduce parameter count in Transformer models?",
            scale=4,
        )


    with gr.Row():
        k_slider = gr.Slider(
            1, 10, value=5, step=1, label="Number of papers to return (K)", scale=2
        )
        summary_check = gr.Checkbox(
            label="Generate GPT summary for each paper", value=False, scale=1
        )
    # TODO: fixed % max width for button? 
    search_btn = gr.Button("Search", variant="primary")
    output_md = gr.Markdown()

    search_btn.click(run_query, inputs=[query_box, k_slider, summary_check], outputs=output_md)

if __name__ == "__main__":
    demo.launch()