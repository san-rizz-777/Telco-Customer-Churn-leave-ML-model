from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys


# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

