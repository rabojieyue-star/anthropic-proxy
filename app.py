import os
from anthropic import Anthropic
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

@app.route("/api/anthropic-proxy", methods=["OPTIONS"])
def anthropic_proxy_options():
    return "", 204

@app.post("/api/anthropic-proxy")
def anthropic_proxy():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set"}), 500
    payload = request.get_json(silent=True) or {}
    model = payload.get("model", "claude-sonnet-4-6")
    system = payload.get("system")
    messages = payload.get("messages")
    max_tokens = payload.get("max_tokens", 1000)
    temperature = payload.get("temperature")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400
    try:
        client = Anthropic(api_key=api_key)
        request_args = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            request_args["system"] = system
        if temperature is not None:
            request_args["temperature"] = temperature
        with client.messages.stream(**request_args) as stream:
            final_message = stream.get_final_message()
        text = "".join(
            block.text for block in final_message.content if getattr(block, "type", None) == "text"
        )
        return jsonify(
            {
                "text": text,
                "usage": {
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                },
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
