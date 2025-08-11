"""
Clean, Modular, General-Purpose Supabase Agent
Handles any type of document with conversation memory and smart routing
"""

import os
import re
import random
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.database_service import DatabaseService
from app.utils.exceptions import AppException

load_dotenv()

class QueryType(Enum):
    SEARCH = "search"
    EDIT = "edit" 
    VIEW = "view"
    ANALYZE = "analyze"
    CHAT = "chat"

class AgentState(TypedDict):
    """Clean state management"""
    messages: Annotated[List[BaseMessage], "Conversation messages"]
    query: str
    query_type: QueryType
    context: Dict[str, Any]
    workspace_id: str
    user_id: str
    filename: Optional[str]
    error: Optional[str]

class SmartSupabaseAgent:
    def __init__(self, workspace_id: str, user_id: str, model: str = None):
        # Core setup
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.db_service = DatabaseService()
        
        # LLM with higher temperature for natural responses
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=model or "llama3-70b-8192",
            temperature=0.3
        )
        
        # Conversation memory for this session
        self.conversation_history = []
        
        # Build workflow
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build clean, simple workflow"""
        workflow = StateGraph(AgentState)
        
        # Core nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("searcher", self._handle_search)
        workflow.add_node("viewer", self._handle_view)
        workflow.add_node("editor", self._handle_edit)
        workflow.add_node("analyzer", self._handle_analyze)
        workflow.add_node("chatter", self._handle_chat)
        
        # Entry point
        workflow.set_entry_point("router")
        
        # Simple routing
        workflow.add_conditional_edges(
            "router",
            self._determine_route,
            {
                "search": "searcher",
                "view": "viewer",
                "edit": "editor", 
                "analyze": "analyzer",
                "chat": "chatter"
            }
        )
        
        # All paths end
        for node in ["searcher", "viewer", "editor", "analyzer", "chatter"]:
            workflow.add_edge(node, END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Smart query routing with context awareness"""
        query = state["query"]
        
        # Build context from conversation history
        context_info = ""
        if self.conversation_history:
            recent_context = self.conversation_history[-3:]  # Last 3 exchanges
            context_info = f"\nRecent conversation context:\n"
            for exchange in recent_context:
                context_info += f"User: {exchange['user']}\nAssistant: {exchange['assistant'][:100]}...\n"
        
        routing_prompt = f"""
You are a smart assistant. Analyze this query and determine the best action.

QUERY: "{query}"
{context_info}

Choose ONE action:
- SEARCH: Find specific information within documents
- VIEW: Show raw file content or specific sections  
- EDIT: Modify, improve, or rewrite content
- ANALYZE: Answer questions about document content, summarize, explain
- CHAT: General conversation, greetings, unclear requests

For document questions like "what is this about?", "tell me about...", "what are the projects?" use ANALYZE.
For finding specific items use SEARCH.
For modifications use EDIT.

Respond with only: SEARCH, VIEW, EDIT, ANALYZE, or CHAT
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=routing_prompt)])
            intent = response.content.strip().upper()
            
            # Map to enum
            intent_map = {
                "SEARCH": QueryType.SEARCH,
                "VIEW": QueryType.VIEW, 
                "EDIT": QueryType.EDIT,
                "ANALYZE": QueryType.ANALYZE,
                "CHAT": QueryType.CHAT
            }
            
            state["query_type"] = intent_map.get(intent, QueryType.CHAT)
            print(f"🧠 Route: {intent} -> {state['query_type']}")
            
        except Exception as e:
            print(f"⚠️ Routing failed: {e}")
            state["query_type"] = QueryType.CHAT
        
        return state
    
    def _determine_route(self, state: AgentState) -> str:
        """Simple route determination"""
        return state["query_type"].value
    
    async def _handle_search(self, state: AgentState) -> AgentState:
        """Handle search requests - fallback to analyze if no vector search"""
        query = state["query"]
        
        # For now, fallback to document analysis since vector search isn't working
        files = await self._get_workspace_files()
        
        if len(files) == 1:
            # Single file - analyze it
            content = files[0]["content"]
            result = await self._smart_analyze(content, query, files[0]["filename"])
        elif len(files) > 1:
            # Multiple files - ask user to specify
            file_list = "\n".join([f"• {f['filename']}" for f in files])
            result = f"I found {len(files)} files. Which one would you like me to search?\n\n{file_list}"
        else:
            result = "No files found in this workspace."
        
        state["messages"].append(AIMessage(content=result))
        return state
    
    async def _handle_view(self, state: AgentState) -> AgentState:
        """Handle view requests"""
        query = state["query"]
        filename = self._extract_filename(query) or state.get("filename")
        
        if not filename:
            files = await self._get_workspace_files()
            if len(files) == 1:
                filename = files[0]["filename"]
            else:
                file_list = "\n".join([f"• {f['filename']}" for f in files])
                result = f"Which file would you like to view?\n\n{file_list}"
                state["messages"].append(AIMessage(content=result))
                return state
        
        file_data = await self._find_file(filename)
        if file_data:
            content = file_data["content"]
            lines = len(content.split('\n'))
            preview = '\n'.join(content.split('\n')[:20])
            result = f"**{filename}** ({lines} lines)\n\n```\n{preview}\n```"
        else:
            result = f"File '{filename}' not found."
        
        state["messages"].append(AIMessage(content=result))
        return state
    
    async def _handle_edit(self, state: AgentState) -> AgentState:
        """Handle edit requests"""
        query = state["query"]
        filename = self._extract_filename(query) or state.get("filename")
        
        if not filename:
            files = await self._get_workspace_files()
            if len(files) == 1:
                filename = files[0]["filename"]
            else:
                file_list = "\n".join([f"• {f['filename']}" for f in files])
                result = f"Which file would you like to edit?\n\n{file_list}"
                state["messages"].append(AIMessage(content=result))
                return state
        
        file_data = await self._find_file(filename)
        if not file_data:
            result = f"File '{filename}' not found."
            state["messages"].append(AIMessage(content=result))
            return state
        
        # Smart editing
        result = await self._smart_edit(file_data, query)
        state["messages"].append(AIMessage(content=result))
        return state
    
    async def _handle_analyze(self, state: AgentState) -> AgentState:
        """Handle analysis requests"""
        query = state["query"]
        filename = self._extract_filename(query) or state.get("filename")
        
        if not filename:
            files = await self._get_workspace_files()
            if len(files) == 1:
                # Single file - analyze it
                file_data = files[0]
                result = await self._smart_analyze(file_data["content"], query, file_data["filename"])
            else:
                file_list = "\n".join([f"• {f['filename']}" for f in files])
                result = f"I found {len(files)} files. Which one would you like me to analyze?\n\n{file_list}"
        else:
            file_data = await self._find_file(filename)
            if file_data:
                result = await self._smart_analyze(file_data["content"], query, filename)
            else:
                result = f"File '{filename}' not found."
        
        state["messages"].append(AIMessage(content=result))
        return state
    
    async def _handle_chat(self, state: AgentState) -> AgentState:
        """Handle general chat"""
        query = state["query"]
        
        # Build context from conversation
        context = ""
        if self.conversation_history:
            context = f"Previous conversation:\n"
            for exchange in self.conversation_history[-2:]:
                context += f"User: {exchange['user']}\nYou: {exchange['assistant']}\n"
        
        chat_prompt = f"""
You are a helpful workspace assistant. Respond naturally and conversationally.

{context}

User: {query}

Keep your response friendly, concise, and helpful. If they're asking about files or documents, let them know what's available in their workspace.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=chat_prompt)])
            result = response.content
        except Exception as e:
            result = "I'm having trouble processing that. Could you try rephrasing?"
        
        state["messages"].append(AIMessage(content=result))
        return state
    
    async def _smart_analyze(self, content: str, query: str, filename: str) -> str:
        """Smart document analysis - works for any document type"""
        
        # Add conversation context
        context_info = ""
        if self.conversation_history:
            context_info = f"\nPrevious conversation context:\n"
            for exchange in self.conversation_history[-2:]:
                context_info += f"User: {exchange['user']}\nAssistant: {exchange['assistant'][:100]}...\n"
        
        # Check if this is asking for specific structured information
        query_lower = query.lower()
        structured_queries = [
            'skills', 'experience', 'projects', 'education', 'qualifications',
            'technologies', 'tools', 'languages', 'frameworks', 'achievements'
        ]
        
        is_structured_query = any(keyword in query_lower for keyword in structured_queries)
        
        if is_structured_query:
            analysis_prompt = f"""
You are extracting specific information from a document. Be direct and well-organized.

DOCUMENT: {filename}
CONTENT:
{content}

USER QUESTION: "{query}"
{context_info}

For this type of question, provide a clean, organized response:
- Use bullet points or clear categories when listing items
- Be direct and factual
- Group related items together
- No extra fluff or conversational padding

Example format for skills:
**Technical Skills:**
• Programming: Python, JavaScript, C++
• Frameworks: React, Django, TensorFlow
• Tools: Docker, Git, AWS

Extract and organize the relevant information clearly.
"""
        else:
            analysis_prompt = f"""
You are a helpful assistant analyzing a document. Be conversational, concise, and direct.

DOCUMENT: {filename}
CONTENT:
{content}

USER QUESTION: "{query}"
{context_info}

Answer their question naturally. Keep it:
- Conversational (like talking to a friend)
- Concise (2-4 sentences usually)
- Direct (answer what they asked)
- No unnecessary jargon or formatting

Just have a normal conversation about the document.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            return response.content
        except Exception as e:
            return f"I had trouble analyzing the document: {str(e)}"
    
    async def _smart_edit(self, file_data: Dict, request: str) -> str:
        """Smart editing for any document type"""
        filename = file_data["filename"]
        content = file_data["content"]
        
        edit_prompt = f"""
You are helping edit a document. The user wants to make changes.

DOCUMENT: {filename}
CURRENT CONTENT:
{content}

USER REQUEST: "{request}"

Based on their request, provide the improved/edited version of the document.
Make the changes they asked for while keeping the document's original purpose and style.

Return ONLY the updated document content - no explanations or prefixes.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = response.content.strip()
            
            # Save the updated content
            await self._save_file(file_data["id"], new_content)
            
            return f"✅ Updated {filename} successfully!"
            
        except Exception as e:
            return f"❌ Edit failed: {str(e)}"
    
    # Helper methods
    async def _get_workspace_files(self) -> List[Dict]:
        """Get all files in workspace"""
        try:
            return await self.db_service.get_workspace_files(self.workspace_id, self.user_id)
        except Exception as e:
            print(f"⚠️ Failed to get files: {e}")
            return []
    
    async def _find_file(self, filename: str) -> Optional[Dict]:
        """Find file by name"""
        files = await self._get_workspace_files()
        
        # Exact match first
        for file in files:
            if file["filename"] == filename:
                return file
        
        # Partial match
        for file in files:
            if filename.lower() in file["filename"].lower():
                return file
        
        return None
    
    def _extract_filename(self, query: str) -> Optional[str]:
        """Extract filename from query"""
        # Look for file extensions
        words = query.split()
        for word in words:
            if '.' in word and any(ext in word.lower() for ext in ['.md', '.txt', '.doc', '.pdf', '.docx', '.py', '.js']):
                return word
        
        # Look for quoted names
        quoted = re.search(r'"([^"]*)"', query)
        if quoted:
            return quoted.group(1)
        
        return None
    
    async def _create_file_version(self, file_id: str, content: str, change_summary: str):
        """Create a version backup of the file"""
        try:
            # Generate unique version number using timestamp (seconds since epoch)
            # This keeps numbers within integer range while still being unique
            timestamp = int(datetime.now().timestamp())  # seconds, not milliseconds
            random_component = random.randint(0, 999)
            version_number = timestamp * 1000 + random_component
            
            version_data = {
                "file_id": file_id,
                "version_number": version_number,
                "content": content,
                "change_summary": change_summary,
                "created_by": self.user_id
            }
            
            result = self.db_service.supabase.table("file_versions").insert(version_data).execute()
            
            if result.data:
                print(f"✅ File version created successfully: {result.data[0]['id']} with version {version_number}")
                return result.data[0]
            else:
                print(f"⚠️ File version creation returned no data")
                print(f"   Version data: {version_data}")
                return None
            
        except Exception as e:
            print(f"⚠️ Failed to create file version: {e}")
            print(f"   File ID: {file_id}")
            print(f"   User ID: {self.user_id}")
            print(f"   Content length: {len(content)}")
            print(f"   Version number attempted: {version_number}")
            # Re-raise the exception so the caller knows it failed
            raise e

    async def _save_file(self, file_id: str, content: str):
        """Save updated file content with version backup"""
        try:
            # Get current file data for backup
            current_file_result = self.db_service.supabase.table("files").select("content").eq("id", file_id).single().execute()
            
            if current_file_result.data:
                current_content = current_file_result.data["content"]
                
                # Create a version backup first
                try:
                    version_result = await self._create_file_version(file_id, current_content, "Agent edit backup")
                    if version_result:
                        print(f"✅ Version backup created before edit: {version_result['id']}")
                    else:
                        print(f"⚠️ Version backup creation returned no result")
                except Exception as version_error:
                    print(f"⚠️ Version backup failed: {version_error}")
                    # Continue with edit even if version creation fails
            
            # Update the file content
            update_data = {
                "content": content,
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.db_service.supabase.table("files").update(update_data).eq("id", file_id).execute()
            
            if result.data:
                print(f"✅ File updated successfully")
                
                # Create a post-edit version to show the new content in history
                try:
                    post_edit_version = await self._create_file_version(file_id, content, "Agent edit result")
                    if post_edit_version:
                        print(f"✅ Post-edit version created: {post_edit_version['id']}")
                except Exception as post_version_error:
                    print(f"⚠️ Post-edit version creation failed: {post_version_error}")
            
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
            raise e
    
    async def chat(self, message: str, filename: str = None) -> str:
        """Main chat interface with memory"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "query": message,
                "query_type": QueryType.CHAT,
                "context": {},
                "workspace_id": self.workspace_id,
                "user_id": self.user_id,
                "filename": filename,
                "error": None
            }
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Get response
            if result["messages"]:
                response = result["messages"][-1].content
            else:
                response = "I'm not sure how to help with that."
            
            # Update conversation memory
            self.conversation_history.append({
                "user": message,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return f"I encountered an error: {str(e)}"

def create_supabase_agent(workspace_id: str, user_id: str, model: str = None) -> SmartSupabaseAgent:
    """Factory function to create agent"""
    return SmartSupabaseAgent(workspace_id, user_id, model)