# agents.py
import os
from openai import OpenAI  # Note: import this way instead of import openai
from typing import List, Dict, Any, Optional, Union
import streamlit as st

class WebSearchTool:
    """A tool for searching the web for information."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for information relevant to the query"
        
    async def run(self, query: str) -> str:
        """
        Simulate web search functionality.
        In a production environment, you would integrate with a real search API.
        """
        # Initialize OpenAI client with project API key
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
        )
        
        try:
            # Use OpenAI to generate a simulated web search result
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a web search tool. Based on the query, provide a concise summary of what web search results might show. Format as if these are search results."},
                    {"role": "user", "content": f"Web search query: {query}"}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Web search error: {str(e)}"

class FileSearchTool:
    """A tool for searching through documents in a vector database."""
    
    def __init__(self, max_num_results: int = 3, vector_store_ids: List[str] = None):
        self.name = "file_search"
        self.description = "Search through stored documents for relevant information"
        self.max_num_results = max_num_results
        self.vector_store_ids = vector_store_ids or []
        
    async def run(self, query: str) -> str:
        """
        Search through vector store for relevant document snippets.
        In a production environment, this would connect to your actual vector database.
        """
        # Initialize OpenAI client with project API key
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
        )
        
        try:
            # This is a simplified simulation of vector store retrieval
            store_ids_str = ", ".join(self.vector_store_ids)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a document retrieval system. Simulate retrieving up to {self.max_num_results} most relevant passages from documents in these vector stores: {store_ids_str}. The passages should be relevant to the query and formatted as search results."},
                    {"role": "user", "content": f"Document search query: {query}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Document search error: {str(e)}"

class Agent:
    """An agent that can use tools to complete a task based on given instructions."""
    
    def __init__(self, name: str, instructions: str, tools: List[Union[WebSearchTool, FileSearchTool]] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        
    async def run(self, message: str) -> Dict[str, Any]:
        """
        Process the user message and use tools as needed.
        """
        # Initialize OpenAI client with project API key
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
        )
        
        # Prepare tools description for the system message
        tools_description = ""
        for tool in self.tools:
            tools_description += f"- {tool.name}: {tool.description}\n"
            
        # Initial system message with instructions and available tools
        messages = [
            {"role": "system", "content": f"{self.instructions}\n\nAvailable tools:\n{tools_description}\n\nWhen you need information, specify which tool to use by saying 'I'll use [tool_name] to find information about [query]'."}
        ]
        
        # Add the user message
        messages.append({"role": "user", "content": message})
        
        # First pass - get the agent's thoughts and tool usage plans
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            agent_response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response})
            
            # Check if the agent wants to use tools
            tool_results = []
            for tool in self.tools:
                # Simple keyword-based detection - could be more sophisticated
                tool_marker = f"I'll use {tool.name}"
                if tool_marker.lower() in agent_response.lower():
                    # Extract the query - this is a simple heuristic
                    try:
                        query_start = agent_response.lower().index(tool_marker.lower())
                        query_text = agent_response[query_start:].split("\n")[0]
                        # Extract the actual query from the text
                        query = query_text.split("about")[1].strip() if "about" in query_text else query_text.split(tool_marker)[1].strip()
                        # Remove any trailing punctuation
                        query = query.rstrip(".!?")
                        
                        # Run the tool
                        tool_result = await tool.run(query)
                        tool_results.append(f"Results from {tool.name}:\n{tool_result}")
                    except Exception as e:
                        tool_results.append(f"Error using {tool.name}: {str(e)}")
            
            # If tools were used, add their results and get final response
            if tool_results:
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"Here are the results from the tools you requested:\n\n{tool_results_text}\n\nPlease provide your final response based on this information."})
                
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7
                )
                
                return {"output": final_response.choices[0].message.content}
            else:
                # No tools were used, return the original response
                return {"output": agent_response}
                
        except Exception as e:
            return {"output": f"Error: {str(e)}"}

class Runner:
    """A class to run agents and return their results."""
    
    @staticmethod
    async def run(agent: Agent, prompt: str):
        """
        Run an agent with the given prompt and return the result.
        """
        result = await agent.run(prompt)
        
        # Create a simple result object with the expected interface
        class Result:
            def __init__(self, output):
                self.final_output = output
                
        return Result(result["output"])
