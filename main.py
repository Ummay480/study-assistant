import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
import chainlit as cl
from typing import Optional

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-1.5-flash"  # Try gemini-1.5-flash; fallback to gemini-pro if needed
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Specified base URL
MAX_INPUT_LENGTH = 1000  # Guardrail: Maximum input length to prevent abuse
ALLOWED_SUBJECTS = ["math", "science", "history", "literature", "biology", "calculus", "physics", "chemistry"]

# Verify API key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

# Set up model with GEMINI_API_KEY
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)

# Guardrail: Function to sanitize input
def sanitize_input(text: str) -> str:
    """Remove potentially harmful characters and excessive whitespace."""
    # Remove HTML tags, scripts, and excessive whitespace
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Guardrail: Function to check if query is study-related
def is_study_related(query: str) -> bool:
    """Check if the query is related to study subjects."""
    query_lower = query.lower()
    return any(subject in query_lower for subject in ALLOWED_SUBJECTS) or "study" in query_lower or "exam" in query_lower

# Specialized agents
question_answer_agent = Agent(
    name="question_answer_agent",
    instructions="You're an expert in answering study-related questions. Provide clear, accurate, and concise answers to questions across subjects like math, science, history, or literature. Use examples where helpful. If the query is not study-related, politely decline to answer and suggest a study-related topic.",
    model=model,
)

summary_agent = Agent(
    name="summary_agent",
    instructions="You're an expert in summarizing study-related content. Summarize provided text or concepts in a concise, easy-to-understand manner, focusing on key points relevant to subjects like math, science, history, or literature. If the content is not study-related, politely decline and suggest a study topic.",
    model=model,
)

practice_question_agent = Agent(
    name="practice_question_agent",
    instructions="You're an expert in creating practice questions. Generate relevant, subject-specific practice questions for subjects like math, science, history, or literature based on the user's request (e.g., 'create math questions' or 'quiz me on biology'). Include answers. If the request is not study-related, politely decline and suggest a study topic.",
    model=model,
)

study_plan_agent = Agent(
    name="study_plan_agent",
    instructions="You're an expert in creating study plans. Generate a detailed, structured study schedule based on the user's goals, subject, and timeframe (e.g., 'study plan for calculus exam in 2 weeks') for subjects like math, science, history, or literature. If the request is not study-related, politely decline and suggest a study topic.",
    model=model,
)

# Main router agent with guardrails
study_assistant = Agent(
    name="study_assistant",
    instructions="""
    Greet the user and ask how you can assist with their study needs for subjects like math, science, history, or literature.
    Route the question to the appropriate sub-agent: question_answer_agent, summary_agent, practice_question_agent, or study_plan_agent.
    If the request is not related to studying or academic subjects, politely inform the user that you can only assist with study-related queries and suggest examples (e.g., 'explain calculus', 'summarize photosynthesis', 'create biology questions', or 'study plan for history').
    If the request is unclear, ask for clarification.
    Do not respond to inappropriate, offensive, or non-academic content; instead, return a polite message redirecting to study topics.
    """,
    model=model,
    handoffs=[question_answer_agent, summary_agent, practice_question_agent, study_plan_agent],
)

# Show greeting on chat start
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="📚 Welcome! I am your Study Assistant!\n\n🤔 How can I assist you with your studies today? (e.g., explain calculus derivatives, summarize photosynthesis, create practice questions for biology, or make a study plan for calculus)").send()

# Handle user questions with guardrails
@cl.on_message
async def handle_message(message: cl.Message):
    # Guardrail: Validate input
    if not message.content or len(message.content) == 0:
        await cl.Message(content="Please provide a study-related question or request.").send()
        return
    if len(message.content) > MAX_INPUT_LENGTH:
        await cl.Message(content=f"Input is too long (max {MAX_INPUT_LENGTH} characters). Please shorten your request.").send()
        return

    # Guardrail: Sanitize input
    sanitized_input = sanitize_input(message.content)

    # Guardrail: Check if query is study-related
    if not is_study_related(sanitized_input):
        await cl.Message(content="Sorry, I can only assist with study-related topics like math, science, history, or literature. Please try something like 'explain calculus' or 'create biology questions'. How can I help you study?").send()
        return

    try:
        # Guardrail: Add rate-limiting placeholder (implement with Chainlit or external library if needed)
        # Example: rate_limit_check(user_id=message.user_id)
        result = await Runner.run(starting_agent=study_assistant, input=sanitized_input)
        
        # Guardrail: Filter response for inappropriate content (basic check)
        if not result.final_output or "error" in result.final_output.lower():
            await cl.Message(content="Sorry, I couldn't process that request. Please try a different study-related question.").send()
            return
        
        await cl.Message(content=result.final_output).send()
    except ValueError as ve:
        await cl.Message(content=f"Invalid input: {str(ve)}. Please provide a valid study-related request.").send()
    except RuntimeError as re:
        await cl.Message(content=f"Processing error: {str(re)}. Please try again or rephrase your request.").send()
    except Exception as e:
        await cl.Message(content="An unexpected error occurred. Please try again or contact support if the issue persists.").send()
