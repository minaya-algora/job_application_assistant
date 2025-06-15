import streamlit as st
import os
import asyncio
from agents import Agent, Runner, FileSearchTool 
from dotenv import load_dotenv

# --- VERY VERY EARLY PRINT ---
print("MAIN_PY_LOG: main.py script is starting/re-running.") 
# --- END OF VERY VERY EARLY PRINT ---

# Streamlit UI Configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Minaya's Application Assistant", layout="wide")

# Load environment variables for local development
load_dotenv(override=True)
print("MAIN_PY_LOG: dotenv loaded.")

# Get vector_store_id from secrets or environment
vector_store_id = st.secrets.get("vector_store_id", os.environ.get("vector_store_id", ""))
print(f"MAIN_PY_LOG: Vector Store ID: '{vector_store_id[:10] if vector_store_id else 'Not Set'}'...")


# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    print("MAIN_PY_LOG: Messages session state initialized.")

# Initialize file search preference (default to True as it's the primary function)
if "use_file_search" not in st.session_state:
    st.session_state.use_file_search = True
    print("MAIN_PY_LOG: use_file_search session state initialized.")

# Read agent instructions from prompt.txt
def get_agent_instructions():
    print("MAIN_PY_LOG: get_agent_instructions called.")
    with open("prompt.txt", "r", encoding="utf-8") as f: 
        return f.read()

# Function to create agent with selected tools
def create_research_assistant():
    print("MAIN_PY_LOG: create_research_assistant called.")
    tools = []
    if st.session_state.use_file_search:
        print(f"MAIN_PY_LOG: Appending FileSearchTool with vector_store_id: '{vector_store_id[:10] if vector_store_id else 'Not Set'}'")
        tools.append(FileSearchTool(
            max_num_results=3,
            vector_store_ids=[vector_store_id] if vector_store_id else [], 
        ))
    
    return Agent(
        name="Minaya's Assistant",
        instructions=get_agent_instructions(),
        tools=tools,
    )

# Async wrapper for running the agent with memory
async def get_research_response(question, history):
    print(f"MAIN_PY_LOG: get_research_response called for question: '{question}'")
    research_assistant = create_research_assistant()
    result = await Runner.run(research_assistant, question) 
    return result.final_output

# Streamlit UI
st.title("🤖 Minaya's Application Assistant")
st.markdown("""
Hi there. I'm here to help you assess Minaya's fit for your organization by giving you the **inside scoop** on her professional experience and work approach — no fluff, no fuss.

I have direct access to her resume and professional documents, so feel free to ask me anything. You can start with simple questions like, *"What's her experience with Zapier?"* or dig deeper with *"How would she handle a project that's off the rails?"*

**Go ahead, put me to the test.**

*Fair warning: I'm in Beta and sometimes hallucinate. Please take my responses with a grain of salt!*
""")

# Sidebar controls
st.sidebar.title("Settings")
st.sidebar.subheader("Document Search")

file_search_active = st.sidebar.checkbox(
    "Search Minaya's Professional Documents (CV, LinkedIn, etc.)", 
    value=st.session_state.use_file_search, 
    key="file_search_toggle"
)

if file_search_active != st.session_state.use_file_search:
    st.session_state.use_file_search = file_search_active
    print("MAIN_PY_LOG: File search toggle changed, rerunning.")
    st.rerun()

st.sidebar.subheader("Conversation")
if st.sidebar.button("Clear Current Conversation"):
    st.session_state.messages = []
    print("MAIN_PY_LOG: Conversation cleared, rerunning.")
    st.rerun()

with st.sidebar.expander("Example Questions"):
    st.markdown("""
    - What are Minaya's key motivations?
    - Describe Minaya's experience with AI-driven marketing.
    - What skills are highlighted in Minaya's resume?
    """) 

st.sidebar.markdown("---")
st.sidebar.markdown("🖤 Made by Minaya Algora")

# Display chat history
# This loop runs on every script rerun, so it's not ideal for one-time logging
# for message in st.session_state.messages:
# print(f"MAIN_PY_LOG: Displaying message from {message['role']}") # Too verbose for here
# with st.chat_message(message["role"]):
# st.markdown(message["content"])

# More targeted message display:
if st.session_state.messages: # Check if there are messages before iterating
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# User input
user_question = st.chat_input("Ask about Minaya's professional profile")
print(f"MAIN_PY_LOG: User input received from chat_input (could be None): '{user_question}'")


if user_question:
    print(f"MAIN_PY_LOG: Processing user_question: '{user_question}'")
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
                
                print(f"MAIN_PY_LOG: Preparing to call get_research_response for question: '{user_question}'")
                st.sidebar.write(f"DEBUG_UI: Calling agent for: {user_question[:30]}...") 
                
                try:
                    response_text = asyncio.run(get_research_response(user_question, st.session_state.messages[:-1]))
                    print(f"MAIN_PY_LOG: get_research_response call completed. Response (start): '{str(response_text)[:100]}'")
                    st.sidebar.write(f"DEBUG_UI: Agent response (start): {str(response_text)[:30]}...")
                except Exception as e_main_run:
                    print(f"MAIN_PY_LOG: EXCEPTION in asyncio.run(get_research_response): {e_main_run}")
                    st.sidebar.error(f"DEBUG_UI_ERROR: Agent call failed: {e_main_run}")
                    response_text = "An error occurred while processing your request with the agent."
                                
                response_placeholder.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    print("MAIN_PY_LOG: No user_question this run.")
