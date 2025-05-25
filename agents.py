# agents.py
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
import streamlit as st
import re 
import time # For polling Assistant run status

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
            print("Warning: FileSearchTool initialized without a valid vector_store_id.")


    async def run(self, query: str) -> str:
        """
        Actually search through the specified OpenAI Vector Store using an Assistant.
        """
        if not self.vector_store_ids or not self.vector_store_ids[0]:
            return "Error: Vector Store ID is not configured for FileSearchTool."

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        vector_store_id = self.vector_store_ids[0] 

        assistant_id_to_use = None
        thread_id_to_use = None

        try:
            assistant_instructions = (
                "You are an AI assistant. Your task is to answer questions based ONLY on the "
                "content of the files provided in the vector store. "
                "If the answer is not found in the documents, clearly state that the information "
                "is not available in the provided documents. Do not make up information or use external knowledge. "
                "When possible, cite the source document implicitly by how you phrase the answer (e.g., 'According to the resume...', 'The cover letter states...')."
            )

            assistant = client.beta.assistants.create(
                name="Minaya Doc Retriever Tool", 
                instructions=assistant_instructions,
                model="gpt-4o", 
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            assistant_id_to_use = assistant.id
            
            thread = client.beta.threads.create()
            thread_id_to_use = thread.id
            
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            
            print(f"LOG_CHECK: FileSearchTool.run - Running Assistant {assistant_id_to_use} on Thread {thread_id_to_use} for query: '{query}'")
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )

            response_content_str = "No text response extracted from assistant." 
            if run.status == 'completed':
                messages_response = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
                response_parts = []
                for msg in messages_response.data:
                    if msg.role == "assistant":
                        for content_block in msg.content:
                            if content_block.type == 'text':
                                response_parts.append(content_block.text.value)
                if response_parts:
                    response_content_str = "\n".join(response_parts)
                else:
                    response_content_str = "The assistant processed the request but provided no textual content."
            else:
                error_message = run.last_error.message if run.last_error else "No specific error details."
                print(f"LOG_CHECK: FileSearchTool.run - Assistant run not completed. Status: {run.status}, Error: {error_message}")
                response_content_str = f"File search failed. Assistant run status: {run.status}. Error: {error_message}"
            
            return response_content_str

        except Exception as e:
            print(f"LOG_CHECK: FileSearchTool.run - Exception: {e}")
            return f"Error during file search with Assistant API: {str(e)}"
        finally:
            if thread_id_to_use:
                try:
                    print(f"LOG_CHECK: FileSearchTool.run - Attempting to delete thread: {thread_id_to_use}")
                    client.beta.threads.delete(thread_id_to_use)
                except Exception as e_del_thread:
                    print(f"Warning: Could not delete temporary thread {thread_id_to_use}: {e_del_thread}")
            if assistant_id_to_use:
                try:
                    print(f"LOG_CHECK: FileSearchTool.run - Attempting to delete assistant: {assistant_id_to_use}")
                    client.beta.assistants.delete(assistant_id_to_use)
                except Exception as e_del_asst:
                    print(f"Warning: Could not delete temporary assistant {assistant_id_to_use}: {e_del_asst}")


class Agent:
    """An agent that uses FileSearchTool to answer questions about Minaya Algora based on given instructions."""
    
    def __init__(self, name: str, instructions: str, tools: List[FileSearchTool] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        
    async def run(self, user_message: str) -> Dict[str, Any]:
        print(f"LOG_CHECK: Agent.run CALLED with user_message: '{user_message}'")

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        file_search_tool = next((t for t in self.tools if isinstance(t, FileSearchTool)), None)

        if not file_search_tool:
            print(f"LOG_CHECK: Agent.run - FileSearchTool not found in tools list.")
            return {"output": "Configuration Error: The FileSearchTool is not available to the agent. Please ensure it's enabled."}
        
        print(f"LOG_CHECK: Agent.run - FileSearchTool instance: {file_search_tool.name}")

        tools_description = f"- **{file_search_tool.name}**: {file_search_tool.description}\n"
            
        agent_system_prompt = f"""
        {self.instructions} 
        You have access to the following tool:
        {tools_description}
        YOUR PROCESS FOR RESPONDING TO QUESTIONS ABOUT MINAYA:
        1. Analyze the User's Question about Minaya.
        2. Plan Tool Usage: You MUST use the '{file_search_tool.name}' tool for any informational request about Minaya.
           State your plan clearly using this exact format: `TOOL_USE: I will use [{file_search_tool.name}] to find information about [specific query for the tool].` (Ensure the tool name is exactly '{file_search_tool.name}' and the query is enclosed in brackets).
        3. Generate Response AFTER Tool Use:
           * The '{file_search_tool.name}' tool will provide a direct answer based on Minaya's documents.
           * Your final response about Minaya should present this information clearly. 
           * If the tool indicates information is not found, relay that message as per your core instructions. Do NOT invent information.
        If the user's message is not an informational question about Minaya (e.g., a simple greeting like 'hello'), you can respond directly.
        However, for any request that requires specific information about Minaya, you must follow the tool usage plan.
        Begin by stating your plan or providing an immediate answer if no tool is needed.
        """

        messages = [{"role": "system", "content": agent_system_prompt}]
        messages.append({"role": "user", "content": user_message})
        
        try:
            print(f"LOG_CHECK: Agent.run - Attempting first LLM call for planning.")
            response1 = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                temperature=0.05, 
                max_tokens=300 
            )
            agent_response_phase1 = response1.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response_phase1})

            print(f"LOG_CHECK: --- Agent Phase 1 Response ---") 
            print(agent_response_phase1)             
            print(f"LOG_CHECK: --- End of Agent Phase 1 Response ---") 

            tool_marker = "TOOL_USE:"
            expected_tool_name_in_brackets_lower = f"[{file_search_tool.name.lower()}]" 
            expected_tool_invocation_start_pattern_lower = f"i will use {expected_tool_name_in_brackets_lower}"
            
            agent_response_lower = agent_response_phase1.lower()

            if tool_marker.lower() in agent_response_lower and \
               expected_tool_invocation_start_pattern_lower in agent_response_lower:
                
                print(f"LOG_CHECK: Agent.run - Found tool marker AND expected tool invocation start pattern.") 
                
                try:
                    # More flexible query extraction:
                    query_intro_marker_lower = "to find information about " # LLM seems to use this phrase
                    
                    query_intro_start_index = agent_response_lower.find(query_intro_marker_lower)

                    if query_intro_start_index != -1:
                        # Start extracting query from after the intro marker in the original case string
                        query_start_actual = query_intro_start_index + len(query_intro_marker_lower)
                        potential_query = agent_response_phase1[query_start_actual:].strip()

                        # Clean up the query: remove leading '[' if present, and trailing ']' or '.'
                        if potential_query.startswith("["):
                            potential_query = potential_query[1:]
                        if potential_query.endswith("]"):
                            potential_query = potential_query[:-1]
                        if potential_query.endswith("."): # Remove trailing period often added by LLM
                            potential_query = potential_query[:-1]
                        
                        query_for_tool = potential_query.strip()

                        # Re-confirm extracted tool name from the original cased string based on found pattern
                        tool_name_phrase_search_start = agent_response_lower.find(expected_tool_invocation_start_pattern_lower)
                        # Get the part after "i will use " which is "[tool_name]"
                        start_of_bracket_in_original_case = tool_name_phrase_search_start + len("i will use ") # length of "i will use "
                        end_of_bracket_in_original_case = agent_response_phase1.find("]", start_of_bracket_in_original_case)
                        
                        extracted_tool_name = "unknown_tool" # Default
                        if end_of_bracket_in_original_case != -1:
                            # Extract from original case string, between the brackets
                            extracted_tool_name = agent_response_phase1[start_of_bracket_in_original_case + 1 : end_of_bracket_in_original_case].strip()

                        print(f"LOG_CHECK: --- Flexible Query Extraction Details ---") 
                        print(f"LOG_CHECK: Extracted tool_name: '{extracted_tool_name}'") 
                        print(f"LOG_CHECK: Extracted query: '{query_for_tool}'")       
                        print(f"LOG_CHECK: --- End of Flexible Query Extraction Details ---")   

                        if extracted_tool_name.lower() == file_search_tool.name.lower() and query_for_tool:
                            print(f"LOG_CHECK: Agent.run - Agent plans to use tool: {file_search_tool.name} with query: '{query_for_tool}'") 
                            tool_output = await file_search_tool.run(query_for_tool)
                            tool_results_text = f"--- Information retrieved by {file_search_tool.name} for query '{query_for_tool}' ---\n{tool_output}\n--- End of Information ---"
                            
                            print(f"LOG_CHECK: Agent.run - Tool output received:\n{tool_output[:200]}...") 

                            final_prompt_messages = messages.copy()
                            final_prompt_messages.append({"role": "user", "content": tool_results_text}) 
                            
                            final_instruction = (
                                f"The '{file_search_tool.name}' tool provided the following information based on Minaya's documents:\n{tool_results_text}\n\n"
                                f"Present this information as your final answer to the original user question: '{user_message}'. "
                                f"Ensure your response aligns with your core instructions (persona, tone, and how to handle missing information if the tool indicated so)."
                            )
                            final_prompt_messages.append({"role": "system", "content": final_instruction})
                            
                            print(f"LOG_CHECK: Agent.run - Attempting second LLM call for final response synthesis.")
                            response2 = client.chat.completions.create(
                                model="gpt-4o",
                                messages=final_prompt_messages,
                                temperature=0.3,
                                max_tokens=700
                            )
                            return {"output": response2.choices[0].message.content}
                        else:
                            print(f"LOG_CHECK: Agent.run - Debug: Tool name or query validity check failed. Extracted Tool: '{extracted_tool_name}', Expected Tool: '{file_search_tool.name.lower()}', Query: '{query_for_tool}'") 
                            return {"output": agent_response_phase1 + "\n\n(Note: Tool name or query was not correctly identified in the plan. Please try rephrasing.)"}
                    else:
                        print(f"LOG_CHECK: Agent.run - Debug: Query intro marker '{query_intro_marker_lower}' not found in plan: '{agent_response_lower}' after finding tool marker and invocation start.") 
                        return {"output": agent_response_phase1 + "\n\n(Note: The plan to find information was not clearly stated as expected. Please try rephrasing.)"}

                except Exception as e:
                    print(f"LOG_CHECK: Agent.run - Error during tool processing (flexible extraction logic): {str(e)}") 
                    return {"output": f"Error during tool processing logic: {str(e)}\nOriginal plan: {agent_response_phase1}"}
            else:
                print(f"LOG_CHECK: Agent.run - Debug: Marker '{tool_marker.lower()}' or pattern '{expected_tool_invocation_start_pattern_lower}' not found in agent response (lower): '{agent_response_lower}'. Agent may not have planned to use a tool.") 
                return {"output": agent_response_phase1} 
        except Exception as e:
            print(f"LOG_CHECK: Agent.run - Error in agent run (main try block): {str(e)}") 
            return {"output": f"Error in agent run: {str(e)}"}


class Runner:
    @staticmethod
    async def run(agent: Agent, prompt: str):
        result = await agent.run(prompt)
        class Result:
            def __init__(self, output):
                self.final_output = output
        return Result(result["output"])
