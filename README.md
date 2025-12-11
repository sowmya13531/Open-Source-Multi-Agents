# Multi-Agent AI System (Open-Source)
## Overview
This project is a multi-agent AI system built using open-source models in Python. It demonstrates coordination between multiple AI agents specialized in research, calculations, finance, and computer vision—all without relying on paid APIs. The system also includes a Gradio-based interactive interface for real-time experimentation.

## Features
1.Research Agent
Searches the web and summarizes articles.
Analyzes information and generates structured reports.
Useful for news, research topics, and content summarization.

## Calculator Agent
Performs arithmetic, factorial, square roots, prime checks, and more.
Handles complex expressions in a single query.
Finance Agent
Fetches stock prices, company profiles, news, and price targets.
Uses open-source financial data tools (like yfinance).

## Vision Agent
Detects objects in images using open-source computer vision models.
Can be extended for real-time image processing.


### Technologies Used
Python 3.12
Gradio (for interactive GUI)
Transformers / Hugging Face (for open-source LLMs)
OpenCV (for vision tasks)
YFinance (for stock and finance data)
DuckDuckGo & Newspaper4k (for web research and article scraping)

**Fully compatible with Google Colab.**

## Installation

Clone the repository:

```bash
git clone https://github.com/sowmya13531/Open-Source-Multi-Agents.git
Open-Source-Multi-Agents
```

### Install dependencies:
```bash
pip install -r requirements.txt
```
***Launch the notebook in Google Colab for interactive execution.***

### Usage
Run the Colab notebook.
Interact with the multi-agent system using the provided Gradio interface.

### Example queries:
# Research
team.run("Tesla stock news")
# Calculator
team.run("factorial(5) + sqrt(16)")
# Finance
team.run("Stock info TSLA")
# Vision
team.run("Detect objects in image")

### Project Highlights

=> Open-source models only: no paid APIs required.
=>Multi-agent coordination: different agents specialize in tasks and can work together.

#### License

This project is licensed under the MIT License.
