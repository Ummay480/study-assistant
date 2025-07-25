import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
import chainlit as cl
import requests
import json

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
MODEL = "gemini-1.5-flash"  # Try gemini-1.5-flash; fallback to gemini-pro if needed
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Specified base URL

# Verify API keys
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")
if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY is not set in the .env file")

# Set up model with GEMINI_API_KEY
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)

# Function to fetch weather data from WeatherAPI (for weather-related study queries)
async def get_weather_data(city, api_key):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}

# Specialized agents
question_answer_agent = Agent(
    name="question_answer_agent",
    instructions="You're an expert in answering study-related questions. Provide clear, accurate, and concise answers to questions across subjects like math, science, history, or literature. Use examples where helpful. If the query involves weather (e.g., meteorology studies), use provided weather data.",
    model=model,
)

summary_agent = Agent(
    name="summary_agent",
    instructions="You're an expert in summarizing content. Summarize provided text or concepts in a concise, easy-to-understand manner, focusing on key points. If weather data is provided, include relevant details for meteorology-related summaries.",
    model=model,
)

practice_question_agent = Agent(
    name="practice_question_agent",
    instructions="You're an expert in creating practice questions. Generate relevant, subject-specific practice questions based on the user's request (e.g., 'create math questions' or 'quiz me on biology'). Include answers. If weather-related, use provided weather data.",
    model=model,
)

study_plan_agent = Agent(
    name="study_plan_agent",
    instructions="You're an expert in creating study plans. Generate a detailed, structured study schedule based on the user's goals, subject, and timeframe (e.g., 'study plan for calculus exam in 2 weeks'). If the subject is meteorology or weather-related, incorporate provided weather data if relevant.",
    model=model,
)

# Main router agent
study_assistant = Agent(
    name="study_assistant",
    instructions="""
    Greet the user and ask how you can assist with their study needs.
    If the query involves a location and weather (e.g., 'study plan for meteorology in London'), extract the city and fetch weather data using the get_weather_data function.
    Route the question to the appropriate sub-agent: question_answer_agent, summary_agent, practice_question_agent, or study_plan_agent, and provide weather data if relevant.
    If the request doesn't match a study-related topic or is unclear, ask for clarification or politely inform the user.
    """,
    model=model,
    handoffs=[question_answer_agent, summary_agent, practice_question_agent, study_plan_agent],
)

# Show greeting on chat start
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="📚 Welcome! I am your Study Assistant!\n\n🤔 How can I assist you with your studies today? (e.g., explain calculus derivatives, summarize photosynthesis, create practice questions for biology, or make a study plan for meteorology in London)").send()

# Handle user questions
@cl.on_message
async def handle_message(message: cl.Message):
    # Extract city from input for weather-related queries
    city = None
    words = message.content.lower().split()
    for i, word in enumerate(words):
        if word in ["in", "for", "at"] and i + 1 < len(words):
            city = " ".join(words[i + 1:i + 3]).capitalize()  # Handle multi-word cities
            break
    # Fetch weather data if city is provided and query is weather-related
    weather_data = await get_weather_data(city, WEATHER_API_KEY) if city and "meteorology" in message.content.lower() else {}
    # Pass input and weather data to the agent
    input_data = f"User input: {message.content}\nWeather data (if relevant): {json.dumps(weather_data)}"
    try:
        result = await Runner.run(starting_agent=study_assistant, input=input_data)
        await cl.Message(content=result.final_output).send()
    except Exception as e:
        await cl.Message(content=f"Sorry, I encountered an error: {str(e)}. Please try again or rephrase your request.").send()