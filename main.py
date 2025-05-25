import streamlit as st
import os
import asyncio
from agents import Agent, Runner, FileSearchTool 
from dotenv import load_dotenv

# Streamlit UI Configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Minaya's SmarterX Application Assistant", layout="wide")

# Load environment variables for local development
load_dotenv(override=True)

# Get vector_store_id from secrets or environment
# USE 'vector_store_id' CONSISTENTLY
vector_store_id = st.secrets.get("vector_store_id", os.environ.get("vector_store_id", ""))

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize file search preference (default to True as it's the primary function)
if "use_file_search" not in st.session_state:
    st.session_state.use_file_search = True

# Read agent instructions from prompt.txt
def get_agent_instructions():
    with open("prompt.txt", "r", encoding="utf-8") as f: 
        return f.read()

# Function to create agent with selected tools
def create_research_assistant():
    tools = []
    if st.session_state.use_file_search:
        tools.append(FileSearchTool(
            max_num_results=3,
            vector_store_ids=[vector_store_id] if vector_store_id else [], # USE 'vector_store_id'
        ))
    
    return Agent(
        name="Minaya's Assistant",
        instructions=get_agent_instructions(),
        tools=tools,
    )

# Async wrapper for running the agent with memory
async def get_research_response(question, history):
    research_assistant = create_research_assistant()
    
    context_messages = []
    if history: 
        for msg in history[-4:]: 
             context_messages.append(f"{msg['role'].capitalize()}: {msg['content']}")
    context_str = "\n".join(context_messages)
    
    result = await Runner.run(research_assistant, question) 
    return result.final_output

# Streamlit UI
st.title("🤖 Minaya's SmarterX Application Assistant")
st.write("Curious about my career goals, my specific expertise in this or that tool, or how I'd fit into your team? Ask me anything about my experience, approach, or potential contributions. I'll do my best to provide the info you need and help you assess my fit for your organization. Please note I'm in Beta and I sometimes hallucinate. Take my responses with a grain of salt!")

# Sidebar controls
st.sidebar.title("Settings")
st.sidebar.subheader("Document Search")

file_search_active = st.sidebar.checkbox(
    "Enable Search in Minaya's Professional Documents", 
    value=st.session_state.use_file_search, 
    key="file_search_toggle"
)

if file_search_active != st.session_state.use_file_search:
    st.session_state.use_file_search = file_search_active
    st.rerun()

st.sidebar.subheader("Conversation")
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

with st.sidebar.expander("Example Questions"):
    st.markdown("""
    - What are Minaya's key motivations?
    - Describe Minaya's experience with AI-driven marketing.
    - What skills are highlighted in Minaya's resume?
    """) # Make sure these are your preferred examples

st.sidebar.markdown("---")
st.sidebar.markdown("🖤 Made by Minaya Algora")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask about Minaya's professional profile")

if user_question:
    # Check if tools list would be empty if file search is disabled
    # A bit more robust: create the assistant once to check its tools
    assistant_for_check = create_research_assistant()
    if not st.session_state.use_file_search and not assistant_for_check.tools:
        st.error("Document Search is currently disabled. Please enable it in the sidebar to ask questions about Minaya.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting documents..."):
                response_placeholder = st.empty()
                response_text = asyncio.run(get_research_response(user_question, st.session_state.messages[:-1]))
                response_placeholder.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
