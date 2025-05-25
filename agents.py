# agents.py
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
import streamlit as st
import re 
import time # For polling Assistant run status

# Initialize OpenAI client globally for the module if preferred, or per function
# For simplicity here, we'll initialize it where needed.
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) # Option for global client

class FileSearchTool:
    """
    A tool that uses OpenAI Assistants API with file_search (retrieval)
    to answer questions based on documents in a specific Vector Store.
    """
    
    def __init__(self, max_num_results: int = 3, vector_store_ids: List[str] = None): # max_num_results not directly used by Assistant API in this way
        self.name = "file_search"
        self.description = (
            "Use this tool to search Minaya Algora's professional documents (Comprehensive Professional Profile, "
            "Cover Letter - SmarterX.pdf, Resume - SmarterX.pdf) for specific details about her experience, "
            "skills, projects, motivations, and qualifications. The tool will provide a direct answer based on the documents."
        )
        self.vector_store_ids = vector_store_ids or []
        if not self.vector_store_ids or not self.vector_store_ids[0]:
            # Log this or handle it; for now, it will likely cause issues later
            print("Warning: FileSearchTool initialized without a valid vector_store_id.")


    async def run(self, query: str) -> str:
        """
        Actually search through the specified OpenAI Vector Store using an Assistant.
        """
        if not self.vector_store_ids or not self.vector_store_ids[0]:
            return "Error: Vector Store ID is not configured for FileSearchTool."

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        vector_store_id = self.vector_store_ids[0] # Use the first ID

        try:
            # Define a simple instruction for the assistant.
            # This assistant is temporary and specific to this query.
            # For production, you might create a more permanent assistant.
            assistant_instructions = (
                "You are an AI assistant. Your task is to answer questions based ONLY on the "
                "content of the files provided in the vector store. "
                "If the answer is not found in the documents, clearly state that the information "
                "is not available in the provided documents. Do not make up information or use external knowledge. "
                "Cite the source document if possible (e.g., 'According to the resume...', 'The cover letter states...')."
            )

            # Create a temporary assistant configured for file search on the specific vector store
            # In a real app, you might manage assistants more persistently
            assistant = client.beta.assistants.create(
                name="Minaya Doc Retriever",
                instructions=assistant_instructions,
                model="gpt-4o", # Or your preferred model that supports Assistants API v2
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            
            # Create a thread
            thread = client.beta.threads.create()
            
            # Add user's query to the thread
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            
            # Run the assistant on the thread
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
                # instructions=assistant_instructions # Instructions can also be overridden here
            )

            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
                response_content = []
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content_block in msg.content:
                            if content_block.type == 'text':
                                response_content.append(content_block.text.value)
                                # You could also look for content_block.text.annotations if you want to see file citations
                
                # Clean up the temporary assistant (optional, but good practice for many temp assistants)
                try:
                    client.beta.assistants.delete(assistant.id)
                except Exception as e_del_asst:
                    print(f"Warning: Could not delete temporary assistant {assistant.id}: {e_del_asst}")
                try:
                    client.beta.threads.delete(thread.id) # Also delete thread
                except Exception as e_del_thread:
                    print(f"Warning: Could not delete temporary thread {thread.id}: {e_del_thread}")

                if response_content:
                    return "\n".join(response_content)
                else:
                    return "The assistant processed the request but provided no text response."
            else:
                # Clean up if run didn't complete successfully
                try:
                    client.beta.assistants.delete(assistant.id)
                except Exception as e_del_asst:
                    print(f"Warning: Could not delete temporary assistant {assistant.id} after run status {run.status}: {e_del_asst}")
                try:
                    client.beta.threads.delete(thread.id) # Also delete thread
                except Exception as e_del_thread:
                    print(f"Warning: Could not delete temporary thread {thread.id} after run status {run.status}: {e_del_thread}")
                return f"File search failed. Assistant run status: {run.status}. Details: {run.last_error or 'No specific error details.'}"

        except Exception as e:
            # Attempt to clean up assistant if created
            if 'assistant' in locals() and assistant and assistant.id:
                try:
                    client.beta.assistants.delete(assistant.id)
                except Exception as e_cleanup:
                    print(f"Error during cleanup of assistant {assistant.id}: {e_cleanup}")
            if 'thread' in locals() and thread and thread.id:
                 try:
                    client.beta.threads.delete(thread.id)
                 except Exception as e_cleanup_thread:
                    print(f"Error during cleanup of thread {thread.id}: {e_cleanup_thread}")
            return f"Error during file search with Assistant API: {str(e)}"


class Agent:
    """An agent that uses FileSearchTool to answer questions about Minaya Algora based on given instructions."""
    
    def __init__(self, name: str, instructions: str, tools: List[FileSearchTool] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        
    async def run(self, user_message: str) -> Dict[str, Any]:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        file_search_tool = next((t for t in self.tools if isinstance(t, FileSearchTool)), None)

        if not file_search_tool:
            return {"output": "Configuration Error: The FileSearchTool is not available to the agent. Please ensure it's enabled."}

        tools_description = f"- **{file_search_tool.name}**: {file_search_tool.description}\n" # Updated description will be used
            
        # The agent_system_prompt might need less emphasis on "simulating" and more on "using the tool's direct answer"
        agent_system_prompt = f"""
        {self.instructions} 

        You have access to the following tool:
        {tools_description}

        YOUR PROCESS FOR RESPONDING TO QUESTIONS ABOUT MINAYA:
        1.  **Analyze the User's Question about Minaya.**
        2.  **Plan Tool Usage:** You MUST use the '{file_search_tool.name}' tool for any informational request about Minaya.
            State your plan clearly using this exact format: `TOOL_USE: I will use [{file_search_tool.name}] to find information about [specific query for the tool].` (Replace brackets).
        3.  **Generate Response AFTER Tool Use:**
            *   The '{file_search_tool.name}' tool will provide a direct answer based on Minaya's documents.
            *   Your final response about Minaya should present this information clearly. 
            *   If the tool indicates information is not found, relay that message as per your core instructions. Do NOT invent information.

        If the user's message is not an informational question about Minaya (e.g., a simple greeting like 'hello'), you can respond directly based on your persona and instructions without using a tool.
        However, for any request that requires specific information about Minaya, you must follow the tool usage plan.

        Begin by stating your plan or providing an immediate answer if no tool is needed.
        """

        messages = [{"role": "system", "content": agent_system_prompt}]
        messages.append({"role": "user", "content": user_message})
        
        try:
            response1 = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=300 # Planning step might not need as many tokens
            )
            agent_response_phase1 = response1.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response_phase1})

            tool_marker = "TOOL_USE:"
            if tool_marker in agent_response_phase1 and f"[{file_search_tool.name}]" in agent_response_phase1:
                try:
                    pattern = re.compile(r"TOOL_USE: I will use \[(?P<tool_name>[^\]]+)\] to find information about \[(?P<query>[^\]]+)\]\.")
                    match = pattern.search(agent_response_phase1)
                    
                    if match and match.group("tool_name").strip() == file_search_tool.name:
                        query_for_tool = match.group("query").strip()
                        
                        print(f"Agent plans to use tool: {file_search_tool.name} with query: '{query_for_tool}'")
                        tool_output = await file_search_tool.run(query_for_tool) # This now calls the RAG version
                        # The tool_output is now expected to be a more complete answer, not just excerpts.
                        tool_results_text = f"--- Information retrieved by {file_search_tool.name} for query '{query_for_tool}' ---\n{tool_output}\n--- End of Information ---"
                        
                        # The second LLM call might now be more about rephrasing or ensuring tone,
                        # rather than synthesizing from raw excerpts. Or it might even be skipped if tool_output is good enough.
                        
                        # For now, let's still do a final pass to ensure adherence to persona and instructions.
                        final_prompt_messages = messages.copy()
                        final_prompt_messages.append({"role": "user", "content": tool_results_text}) 
                        
                        final_instruction = (
                            f"The '{file_search_tool.name}' tool provided the following information based on Minaya's documents:\n{tool_results_text}\n\n"
                            f"Present this information as your final answer to the original user question: '{user_message}'. "
                            f"Ensure your response aligns with your core instructions (persona, tone, and how to handle missing information if the tool indicated so)."
                        )
                        final_prompt_messages.append({"role": "system", "content": final_instruction})
                        
                        response2 = client.chat.completions.create(
                            model="gpt-4o",
                            messages=final_prompt_messages,
                            temperature=0.3,
                            max_tokens=700 # Allow more tokens for the final answer
                        )
                        return {"output": response2.choices[0].message.content}
                    else:
                        return {"output": agent_response_phase1 + "\n\n(Note: I planned to use the file search tool but couldn't parse the details correctly or referred to an incorrect tool. Please try rephrasing.)"}
                except Exception as e:
                    return {"output": f"Error during tool processing: {str(e)}\nOriginal plan: {agent_response_phase1}"}
            else:
                return {"output": agent_response_phase1} # If no tool planned
        except Exception as e:
            return {"output": f"Error in agent run: {str(e)}"}


class Runner:
    @staticmethod
    async def run(agent: Agent, prompt: str):
        result = await agent.run(prompt)
        class Result:
            def __init__(self, output):
                self.final_output = output
        return Result(result["output"])
