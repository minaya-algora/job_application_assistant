# agents.py
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
import streamlit as st
import re # For parsing tool usage

class FileSearchTool:
    """A tool for searching through Minaya Algora's professional documents."""
    
    def __init__(self, max_num_results: int = 3, vector_store_ids: List[str] = None):
        self.name = "file_search"
        self.description = (
            "Use this tool to search Minaya Algora's professional documents (Comprehensive Professional Profile, "
            "Cover Letter - SmarterX.pdf, Resume - SmarterX.pdf) for specific details about her experience, "
            "skills, projects, motivations, and qualifications."
        )
        self.max_num_results = max_num_results
        # self.vector_store_ids = vector_store_ids or [] # Stored but not directly used in this simulation's API call logic
        
    async def run(self, query: str) -> str:
        """
        Simulate searching through Minaya's professional documents.
        """
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        try:
            # Refined system prompt for more specific simulation
            system_prompt = (
                f"You are simulating a search through Minaya Algora's professional documents. These documents include her "
                f"'Comprehensive Professional Profile', 'Cover Letter - SmarterX.pdf', and 'Resume - SmarterX.pdf'.\n"
                f"The user is asking about: '{query}'.\n"
                f"Based on this query, retrieve up to {self.max_num_results} relevant-sounding passages that could plausibly be "
                f"extracted from these specific types of documents.\n"
                f"- If the query is about Minaya's overall motivations, personality, career summary, or unique value proposition, "
                f"provide excerpts that sound like they come from the 'Comprehensive Professional Profile' or 'Cover Letter - SmarterX.pdf'.\n"
                f"- If the query is about specific skills, job experiences, project details, or quantifiable achievements, "
                f"provide excerpts that sound like they come from the 'Resume - SmarterX.pdf'.\n"
                f"- If the query relates to her interest in a specific company (e.g., SmarterX), "
                f"provide excerpts that sound like they come from the 'Cover Letter - SmarterX.pdf'.\n"
                f"Format each passage clearly. Start each passage with a phrase indicating the likely source document, for example: "
                f"'From Minaya's Comprehensive Professional Profile:', 'From her Resume (SmarterX):', or 'From her Cover Letter (SmarterX):'.\n"
                f"If no plausible information matching the query would typically be found in these types of documents, respond with: "
                f"'No specific information found in Minaya's documented materials for this query.'"
            )
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Find information relevant to: {query}"}
                ],
                max_tokens=450,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Document search simulation error: {str(e)}"

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

        tools_description = f"- **{file_search_tool.name}**: {file_search_tool.description}\n"
            
        agent_system_prompt = f"""
        {self.instructions} 

        You have access to the following tool:
        {tools_description}

        YOUR PROCESS FOR RESPONDING TO QUESTIONS ABOUT MINAYA:
        1.  **Analyze the User's Question about Minaya.**
        2.  **Plan Tool Usage:** You MUST use the '{file_search_tool.name}' tool for any informational request about Minaya.
            State your plan clearly using this exact format: `TOOL_ ഇന്ത്യ I will use [{file_search_tool.name}] to find information about [specific query for the tool].` (Replace brackets).
        3.  **Generate Response AFTER Tool Use:**
            *   Your final answer about Minaya MUST be based **SOLELY** on the information retrieved by the '{file_search_tool.name}' tool and your core instructions.
            *   If the retrieved information from '{file_search_tool.name}' doesn't answer the question, you MUST state that the information is not in Minaya's documented materials. Do NOT invent information.

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
                max_tokens=500
            )
            agent_response_phase1 = response1.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response_phase1})

            tool_marker_prefix = "TOOL_ ഇന്ത്യ I will use "
            if tool_marker_prefix in agent_response_phase1 and f"[{file_search_tool.name}]" in agent_response_phase1:
                try:
                    pattern = re.compile(r"TOOL_ ഇന്ത്യ I will use \[(?P<tool_name>[^\]]+)\] to find information about \[(?P<query>[^\]]+)\]\.")
                    match = pattern.search(agent_response_phase1)
                    
                    if match and match.group("tool_name").strip() == file_search_tool.name:
                        query_for_tool = match.group("query").strip()
                        
                        print(f"Agent plans to use tool: {file_search_tool.name} with query: '{query_for_tool}'")
                        tool_output = await file_search_tool.run(query_for_tool)
                        tool_results_text = f"--- Results from {file_search_tool.name} for query '{query_for_tool}' ---\n{tool_output}\n--- End of Results ---"
                        
                        final_prompt_messages = messages.copy()
                        final_prompt_messages.append({"role": "user", "content": tool_results_text})
                        
                        final_instruction = (
                            f"You previously received the following excerpts from Minaya's documents after using the '{file_search_tool.name}' tool:\n{tool_results_text}\n\n"
                            f"Based **ONLY** on these provided document excerpts and your core instructions (especially about what to do if information is missing), "
                            f"now provide your final answer to the original user question: '{user_message}'. "
                            "Do not use any other knowledge or invent details. If the excerpts do not contain the specific information, "
                            "you MUST state that, as per your instructions (e.g., 'Based on Minaya's documented materials, I don't have that specific information.')."
                        )
                        final_prompt_messages.append({"role": "system", "content": final_instruction})
                        
                        response2 = client.chat.completions.create(
                            model="gpt-4o",
                            messages=final_prompt_messages,
                            temperature=0.3,
                            max_tokens=500
                        )
                        return {"output": response2.choices[0].message.content}
                    else:
                        return {"output": agent_response_phase1 + "\n\n(Note: I planned to use the file search tool but couldn't parse the details correctly or referred to an incorrect tool. Please try rephrasing.)"}
                except Exception as e:
                    return {"output": f"Error during tool processing: {str(e)}\nOriginal plan: {agent_response_phase1}"}
            else:
                return {"output": agent_response_phase1}
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
