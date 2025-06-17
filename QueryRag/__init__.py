import azure.functions as func
import json, logging
from transformers import pipeline

hf_generator = pipeline(
    "text-generation",
    model=""
)
from src.cli import query as query_func
'''
azure function entrypoint. registers route hanlder by using decorator api to expose POST endpoint at /query 
'''

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.function_name(name="query_rag")
@app.route(route="query", methods=["POST"])
def run_query(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        q = body.get("query")
        top = int(body.get("top", 5))
        if not q:
            return func.HttpResponse("Missing 'query' field", status_code=400)
        result = query_func.callback(q, top) # typer callback to cli.py returns None and prints to std. TODO: return and use? 
        return func.HttpResponse(json.dumps({"ok": True}), mimetype="application/json")
    except Exception as e:
        logging.exception("Query failed")
        return func.HttpResponse(str(e), status_code=500)