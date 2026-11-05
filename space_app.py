"""Entry point for a Hugging Face Space with the Gradio SDK (free CPU hardware).

Runs server.py, which serves both apps from one address: the HTML/CSS app at /  and the
Gradio app at /gradio/. Spaces set GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (0.0.0.0:7860).
"""
import os

import uvicorn

import server

if __name__ == "__main__":
    uvicorn.run(server.api, host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
