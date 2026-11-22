import os
import certifi
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# Load Environment Variables
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# 1. Tools
search_tool = TavilySearch(max_results=2)

@tool
def get_weather(city: str) -> str:
    """Fetch the current weather of a specific city using the OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY") or os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key is not available."

    url = "https://openweathermap.org"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"OpenWeatherMap API error: {data.get('message', 'Unknown error')}"

        location = data["name"]
        country = data["sys"]["country"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"].capitalize()

        return (
            f"Current weather in {location}, {country}: {desc}. "
            f"Temperature: {temp}°C (feels like {feels_like}°C), Humidity: {humidity}%."
        )
    except Exception as e:
        return f"Failed to fetch weather data: {str(e)}"

# 2. LLM & Agent
llm = init_chat_model(
    "gemini-3.1-flash-lite",
    model_provider="google_genai",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather],
    system_prompt=(
        "You are a helpful AI assistant. Use get_weather for real-time city weather queries "
        "and TavilySearch for general, up-to-date information."
    ),
)

# 3. FastAPI App & Pydantic Models
app = FastAPI(title="Weather & Search Agent API")

class QueryRequest(BaseModel):
    prompt: str = Field(..., example="What is the weather in Ranchi?")

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"status": "running", "docs": "http://127.0.0"}

@app.post("/ask", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": request.prompt}]
        })
        raw_content = result["messages"][-1].content

        # Extract plain text since Gemini returns a list of content blocks
        if isinstance(raw_content, list):
            answer_text = "".join(
                block["text"] for block in raw_content if block.get("type") == "text"
            )
        else:
            answer_text = str(raw_content)

        return QueryResponse(answer=answer_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
