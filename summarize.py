import os
import gradio as gr
import torch  # Setup PyTorch for Machine Learning
import datasets  # set up local access to hugging face model
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())
hf_api_key = os.getenv("hf_api_key")
print(hf_api_key)

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize(input_text):  # defines summarize
    output = get_completion(input_text)
    return output[0]['summary_text']


gr.close_all()
demo = gr.Interface(fn=summarize, inputs="text", outputs="text")  # gradio inputs and the function
demo.launch()  # Starts gradio interface
