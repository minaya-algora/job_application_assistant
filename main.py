import streamlit as st
import os
import asyncio
from agents import Agent, Runner, WebSearchTool, FileSearchTool # Your custom agent modules
from dotenv import load_dotenv
# import openai # Not strictly needed in main.py if agents.py handles client init

# Streamlit UI Configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Minaya's SmarterX Application Assistant", layout="wide") # This is the one and only call

# --- TEMPORARY DEBUG SECTION ---
# This will print to your Streamlit app's UI when deployed.
# REMEMBER TO REMOVE THIS AFTER DEBUGGING.
if "OPENAI_API_KEY" in st.secrets:
    key_from_secrets = st.secrets["OPENAI_API_KEY"]
    st.write(f"Debug: Key from secrets starts with: {key_from_secrets[:12]} and ends with: {key_from_secrets[-6:]}")
    st.write(f"Debug: Length of key from secrets: {len(key_from_secrets)}")
else:
    st.write("Debug: OPENAI_API_KEY not found in st.secrets!")
# --- END OF TEMPORARY DEBUG SECTION ---

# Load environment variables for local development
load_dotenv(override=True)

# Get API keys (prioritize Streamlit secrets if available)
# Note: OPENAI_API_KEY variable here isn't directly used if agents.py re-fetches from st.secrets
# vector_store_id is used in create_research_assistant
current_vector_store_id = "" # Initialize
if "OPENAI_API_KEY" in st.secrets: # This check is fine for knowing if the secret exists
    # OPENAI_API_KEY_main = st.secrets["OPENAI_API_KEY"] # Renamed to avoid confusion, not used by agents
    current_vector_store_id = st.secrets.get("vector_store_id", "")
else:
    # Fallback to environment variables for local development
    # OPENAI_API_KEY_main = os.environ.get("OPENAI_API_KEY", "") # Renamed, not used by agents
    current_vector_store_id = os.environ.get("vector_store_id", "")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize search tool preferences if they don't exist
if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = True
if "use_file_search" not in st.session_state:
    st.session_state.use_file_search = True

# Read agent instructions from prompt.txt
def get_agent_instructions():
    with open("prompt.txt", "r") as f:
        return f.read()

# Function to create agent with selected tools
def create_research_assistant():
    tools = []

    
    
    if st.session_state.use_web_search:
        tools.append(WebSearchTool()) # Assumes WebSearchTool uses st.secrets["OPENAI_API_KEY"]
        
    if st.session_state.use_file_search:
        # Pass the vector_store_id fetched earlier
        tools.append(FileSearchTool(
            max_num_results=3,
            vector_store_ids=[current_vector_store_id] if current_vector_store_id else [],
        ))
    
    return Agent(
        name="Minaya's Assistant"
        instructions=get_agent_instructions(),
        tools=tools,
    )

# Async wrapper for running the agent with memory
async def get_research_response(question, history):
    # Create agent with current tool selections
    research_assistant = create_research_assistant()
    
    # Combine history and current question to provide context
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = f"Context of our conversation:\n{context}\n\nCurrent question: {question}"
    
    result = await Runner.run(research_assistant, prompt)
    return result.final_output

# Streamlit UI starts here (after set_page_config and debug)
st.title("🤖 Minaya's SmarterX Application Assistant")
st.write("Curious about my career goals, my specific expertise in this or that tool, or how I'd fit into your team? Ask me anything about my experience, approach, or potential contributions. I'll do my best to provide the info you need and help you assess my fit for your organization. Please note I'm in Beta and I sometimes hallucinate. Take my responses with a grain of salt!")

# Sidebar controls for search tool selection
st.sidebar.title("Search Settings")

# Tool selection toggles
st.sidebar.subheader("Select Search Sources")
web_search = st.sidebar.checkbox("Web Search", value=st.session_state.use_web_search, key="web_search_toggle")
file_search = st.sidebar.checkbox("Vector Store Search", value=st.session_state.use_file_search, key="file_search_toggle")

# Update session state when toggles change
if web_search != st.session_state.use_web_search:
    st.session_state.use_web_search = web_search
    
if file_search != st.session_state.use_file_search:
    st.session_state.use_file_search = file_search

# Validate that at least one search source is selected
if not st.session_state.use_web_search and not st.session_state.use_file_search:
    st.sidebar.warning("Please select at least one search source")

# Conversation controls
st.sidebar.subheader("Conversation")
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun() # Changed from experimental_rerun to rerun

# Display some helpful examples
with st.sidebar.expander("Example Questions"):
    st.markdown("""
    - What's your expertise with automation tools like Zapier or Make?
    - What does your ideal work environment look like?
    - What makes you stand out from other marketing professionals?
    - What personality traits do you have?
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🖤 Made by Minaya Algora")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask your research question")

if user_question:
    # Check if at least one search source is selected
    if not st.session_state.use_web_search and not st.session_state.use_file_search:
        st.error("Please select at least one search source in the sidebar")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                response_placeholder = st.empty()
                
                # Get response from agent
                response = asyncio.run(get_research_response(user_question, st.session_state.messages))
                
                # Update response placeholder
                response_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
