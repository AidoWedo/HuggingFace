import gradio as gr
import os
import torch  # Setup PyTorch for Machine Learning
import datasets  # set up local access to hugging face model
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())
hf_api_key = os.getenv("hf_api_key")
print(hf_api_key)

gr.load("models/openai/whisper-large-v2").launch()

