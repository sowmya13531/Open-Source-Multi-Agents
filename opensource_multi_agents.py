from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()  # Choose your image

import nltk
nltk.download('punkt')

import math
from duckduckgo_search import DDGS
from newspaper import Article
from transformers import pipeline
import gradio as gr
import yfinance as yf
from ultralytics import YOLO
from googlesearch import search
import cv2


# ------------------- LLM Research Agent -------------------
class LLMResearchAgent:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct", device=0):
        # Fully open-source, no login required
        print("Loading Falcon-7B model (this may take a while)...")
        self.generator = pipeline("text-generation", model=model_name, device=device)

    def summarize(self, text):
        if not text.strip():
            return "No text to summarize."
        output = self.generator(f"Summarize this:\n{text}", max_length=512, do_sample=True)
        return output[0]["generated_text"]

# ------------------- Web Agent -------------------
class WebAgent:
    def search(self, query, max_results=5):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                # Extract URLs only
                urls = [r['href'] for r in results]
            return urls
        except:
            return ["Web search failed"]


# ------------------- News Agent -------------------
class NewsAgent:
    def read_article(self, url, max_chars=1000):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text[:max_chars]
        except:
            return None

# ------------------- Calculator Agent -------------------
class CalculatorAgent:
    def calculate(self, expression):
        try:
            return eval(expression, {"__builtins__": {}}, math.__dict__)
        except Exception as e:
            return str(e)

# ------------------- Finance Agent -------------------
class FinanceAgent:
    def get_stock_info(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "name": info.get("shortName"),
                "price": info.get("currentPrice"),
                "sector": info.get("sector"),
                "marketCap": info.get("marketCap")
            }
        except:
            return {"error": "Failed to fetch stock info"}

# ------------------- Vision Agent -------------------
class VisionAgent:
    def __init__(self, model_path="yolov8n.pt"):
        # Use uploaded file or default YOLO model
        self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        try:
            results = self.model(image_path)
            return results.pandas().xyxy[0]  # Pandas DataFrame
        except:
            return "Object detection failed"

# ------------------- Team Agent -------------------
class AgentTeam:
    def __init__(self):
        self.research_agent = LLMResearchAgent()
        self.web_agent = WebAgent()
        self.news_agent = NewsAgent()
        self.calculator_agent = CalculatorAgent()
        self.finance_agent = FinanceAgent()
        self.vision_agent = VisionAgent()

    def run(self, query):
        query_lower = query.lower()
        # Routing logic
        if any(op in query_lower for op in ["+", "-", "*", "/", "factorial", "sqrt"]):
            return self.calculator_agent.calculate(query)
        elif any(tick in query_lower.upper() for tick in ["AAPL","TSLA","GOOG","MSFT"]):
            ticker = query.split()[-1].upper()
            return self.finance_agent.get_stock_info(ticker)
        elif "image" in query_lower or "detect" in query_lower:
            # Use first uploaded image
            image_path = list(uploaded.keys())[0]
            return self.vision_agent.detect_objects(image_path)
        else:
            urls = self.web_agent.search(query)
            articles_text = "\n".join([self.news_agent.read_article(url) for url in urls if self.news_agent.read_article(url)])
            return self.research_agent.summarize(articles_text)

