import streamlit as st
import os
import asyncio
from agents import Agent, Runner, FileSearchTool # WebSearchTool removed
from dotenv import load_dotenv

# Streamlit UI Configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Minaya's SmarterX Application Assistant", layout="wide")

# Load environment variables for local development
load_dotenv(override=True)

# Get vector_store_id from secrets or environment
current_vector_store_id = st.secrets.get("vector_store_id", os.environ.get("vector_store_id", ""))

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize file search preference (default to True as it's the primary function)
if "use_file_search" not in st.session_state:
    st.session_state.use_file_search = True

# Read agent instructions from prompt.txt
def get_agent_instructions():
    with open("prompt.txt", "r", encoding="utf-8") as f: # Added encoding
        return f.read()

# Function to create agent with selected tools
def create_research_assistant():
    tools = []
    if st.session_state.use_file_search:
        tools.append(FileSearchTool(
            max_num_results=3,
            vector_store_ids=[current_vector_store_id] if current_vector_store_id else [],
        ))
    
    return Agent(
        name="Minaya's Assistant",
        instructions=get_agent_instructions(),
        tools=tools,
    )

# Async wrapper for running the agent with memory
async def get_research_response(question, history):
    research_assistant = create_research_assistant()
    
    # Combine history and current question to provide context
    # For this focused agent, history might be less critical for the tool-using step,
    # but good to keep for conversational flow if the LLM can handle it.
    context_messages = []
    if history: # Include some recent history if available
        for msg in history[-4:]: # Last 4 messages to keep context concise
             context_messages.append(f"{msg['role'].capitalize()}: {msg['content']}")
    context_str = "\n".join(context_messages)
    
    # The prompt to the agent's .run() method is just the user's current question.
    # The agent's internal system prompt handles its core instructions and tool usage logic.
    # The history is used here mainly for the final display and could be used by the LLM if it sees the full message list.
    
    result = await Runner.run(research_assistant, question) # Pass only current question
    return result.final_output

# Streamlit UI
st.title("🤖 Minaya's SmarterX Application Assistant")
st.write("Curious about my career goals, my specific expertise in this or that tool, or how I'd fit into your team? Ask me anything about my experience, approach, or potential contributions. I'll do my best to provide the info you need and help you assess my fit for your organization. Please note I'm in Beta and I sometimes hallucinate. Take my responses with a grain of salt!")

# Sidebar controls
st.sidebar.title("Settings")
st.sidebar.subheader("Document Search")

file_search_active = st.sidebar.checkbox(
    "Enable Search in Minaya's Professional Docs", 
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
    - What experience does Minaya have with automation tools like Zapier, Make, or n8n?
    - What does Minaya's ideal work environment look like?
    - What makes Minaya stand out from other marketing professionals?
    - What personality traits does Minaya have?
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("🖤 Made by Minaya Algora")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask about Minaya's professional profile")

if user_question:
    if not st.session_state.use_file_search and not create_research_assistant().tools:
        st.error("Document Search is currently disabled. Please enable it in the sidebar to ask questions about Minaya.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting documents..."):
                response_placeholder = st.empty()
                # Pass current user_question and relevant history to the agent
                response_text = asyncio.run(get_research_response(user_question, st.session_state.messages[:-1])) # Pass history excluding current q
                response_placeholder.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
