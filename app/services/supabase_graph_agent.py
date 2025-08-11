"""
Supabase-enabled Smart Graph-Based Agent using LangGraph
Analyzes queries intelligently and routes to appropriate tools
Works with Supabase database for file operations instead of local files
"""

import os
import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from pinecone import Pinecone
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
    """State that flows through the graph nodes"""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    conversation_history: List[Dict[str, str]]  # Session memory
    query: str
    query_type: QueryType
    confidence: float
    context: Dict[str, Any]  # General context instead of entities
    search_results: List[Dict]
    file_content: Optional[str]
    file_id: Optional[str]
    filename: Optional[str]
    workspace_id: str
    user_id: str
    next_action: str
    error: Optional[str]
    document_type: Optional[str]  # Auto-detected document type
    session_context: Dict[str, Any]  # Session-level context

class SupabaseGraphAgent:
    def __init__(self, workspace_id: str, user_id: str, model: str = None):
        # Initialize LLM with configurable model
        selected_model = model or "llama3-70b-8192"  # Default model
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=selected_model,
            temperature=0.7
        )
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index = self.pc.Index("docupilot")
        except Exception as e:
            print(f"⚠️ Pinecone initialization failed: {e}")
            self.pc = None
            self.index = None
        
        # Database service for Supabase operations
        self.db_service = DatabaseService()
        
        # Workspace and user context
        self.workspace_id = workspace_id
        self.user_id = user_id
        
        # Session memory for conversation context
        self.conversation_memory = []
        self.session_context = {}
        
        # Create tools
        self.tools = self._create_tools()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _create_tools(self) -> List[StructuredTool]:
        """Create Supabase-enabled tools"""
        
        class SearchInput(BaseModel):
            query: str = Field(description="Search query")
            top_k: int = Field(default=5, description="Number of results")
        
        class ViewInput(BaseModel):
            filename: str = Field(description="Filename to view")
            start_line: int = Field(default=None, description="Start line (optional)")
            end_line: int = Field(default=None, description="End line (optional)")
        
        class EditInput(BaseModel):
            filename: str = Field(description="Filename to edit")
            new_content: str = Field(description="New content for the file")
        
        return [
            StructuredTool.from_function(
                func=self._search_tool,
                name="search",
                description="Search documents for relevant content",
                args_schema=SearchInput
            ),
            StructuredTool.from_function(
                func=self._view_tool,
                name="view",
                description="View file content from Supabase",
                args_schema=ViewInput
            ),
            StructuredTool.from_function(
                func=self._edit_tool,
                name="edit",
                description="Edit file content in Supabase",
                args_schema=EditInput
            )
        ]
    
    async def _get_workspace_files(self) -> List[Dict]:
        """Get all files in the current workspace"""
        try:
            files = await self.db_service.get_workspace_files(self.workspace_id, self.user_id)
            return files
        except Exception as e:
            print(f"⚠️ Failed to get workspace files: {e}")
            return []
    
    async def _find_file_by_name(self, filename: str) -> Optional[Dict]:
        """Find a file by name in the current workspace"""
        try:
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
        except Exception as e:
            print(f"⚠️ Failed to find file: {e}")
            return None
    
    def _search_tool(self, query: str, top_k: int = 5) -> str:
        """Search tool using Pinecone (same as before)"""
        if not self.index:
            return "❌ Search unavailable - no vector index"
        
        try:
            # Get available namespaces based on workspace files
            namespaces = self._get_workspace_namespaces()
            if not namespaces:
                return "📄 No documents indexed in this workspace"
            
            all_results = []
            for ns in namespaces:
                try:
                    results = self.index.search(
                        namespace=ns,
                        query={"inputs": {"text": query}, "top_k": max(1, top_k // len(namespaces))},
                        fields=["chunk_text", "filename", "start_line", "end_line", "section_path"]
                    )
                    hits = results.get("result", {}).get("hits", [])
                    for hit in hits:
                        hit["namespace"] = ns
                    all_results.extend(hits)
                except Exception as e:
                    continue
            
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return self._format_search_results(all_results[:top_k], query)
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    async def _view_tool(self, filename: str, start_line: int = None, end_line: int = None) -> str:
        """View file content from Supabase"""
        try:
            file_data = await self._find_file_by_name(filename)
            if not file_data:
                return f"❌ File not found: {filename}"
            
            content = file_data["content"]
            lines = content.split('\n')
            
            if start_line is None and end_line is None:
                # Show file overview
                total_lines = len(lines)
                preview = '\n'.join(lines[:20])
                return f"📄 **{filename}** ({total_lines} lines)\n```\n{preview}\n```"
            
            start_idx = max(0, (start_line or 1) - 1)
            end_idx = min(len(lines), end_line or len(lines))
            section_content = '\n'.join(lines[start_idx:end_idx])
            
            return f"📄 **{filename}** (lines {start_idx+1}-{end_idx})\n```\n{section_content}\n```"
            
        except Exception as e:
            return f"❌ View error: {str(e)}"
    
    async def _edit_tool(self, filename: str, new_content: str) -> str:
        """Edit file content in Supabase"""
        try:
            file_data = await self._find_file_by_name(filename)
            if not file_data:
                return f"❌ File not found: {filename}"
            
            file_id = file_data["id"]
            
            # Create a version backup first
            try:
                version_result = await self._create_file_version(file_id, file_data["content"], "Agent edit backup")
                if version_result:
                    print(f"✅ Version backup created before edit: {version_result['id']}")
                else:
                    print(f"⚠️ Version backup creation returned no result")
            except Exception as version_error:
                print(f"⚠️ Version backup failed: {version_error}")
                # Continue with edit even if version creation fails
            
            # Update the file content
            update_data = {
                "content": new_content,
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.db_service.supabase.table("files").update(update_data).eq("id", file_id).execute()
            
            if result.data:
                print(f"✅ File updated successfully: {filename}")
                
                # Create a post-edit version to show the new content in history
                try:
                    post_edit_version = await self._create_file_version(file_id, new_content, "Agent edit result")
                    if post_edit_version:
                        print(f"✅ Post-edit version created: {post_edit_version['id']}")
                except Exception as post_version_error:
                    print(f"⚠️ Post-edit version creation failed: {post_version_error}")
                
                return f"✅ **Edit completed successfully!**\n\n**File:** {filename}\n**Workspace:** {self.workspace_id}\n\n**Changes made:** The document has been updated in the database."
            else:
                return f"❌ Failed to update file in database"
            
        except Exception as e:
            print(f"❌ Edit tool error: {str(e)}")
            return f"❌ Edit error: {str(e)}"
    
    async def _create_file_version(self, file_id: str, content: str, change_summary: str):
        """Create a version backup of the file"""
        try:
            # Generate unique version number using timestamp + random component
            import random
            timestamp = int(datetime.now().timestamp() * 1000)  # milliseconds
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
    
    def _get_workspace_namespaces(self) -> List[str]:
        """Get available document namespaces for the workspace"""
        # This would need to be implemented based on how you index workspace files
        # For now, return a default namespace based on workspace_id
        return [f"workspace_{self.workspace_id}"]
    
    def _format_search_results(self, hits: List[Dict], query: str) -> str:
        """Format search results (same as before)"""
        if not hits:
            return f"🔍 No results for: '{query}'"
        
        results = [f"🔍 **Search Results for:** '{query}'\n"]
        
        for i, hit in enumerate(hits, 1):
            fields = hit.get("fields", {})
            filename = fields.get("filename", "unknown")
            start_line = fields.get("start_line", -1)
            end_line = fields.get("end_line", -1)
            content = fields.get("chunk_text", "").strip()[:200] + "..."
            
            results.append(f"**{i}.** {filename} (lines {start_line}-{end_line})\n{content}\n")
        
        return "\n".join(results)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes - all async now
        workflow.add_node("analyzer", self._analyze_query)
        workflow.add_node("searcher", self._search_node)
        workflow.add_node("viewer", self._view_node)
        workflow.add_node("editor", self._edit_node)
        workflow.add_node("document_analyzer", self._analyze_document_node)
        workflow.add_node("responder", self._respond_node)
        
        # Set entry point
        workflow.set_entry_point("analyzer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyzer",
            self._route_query,
            {
                "search": "searcher",
                "view": "viewer", 
                "edit": "editor",
                "analyze": "document_analyzer",
                "respond": "responder"
            }
        )
        
        # All nodes lead to responder
        workflow.add_edge("searcher", "responder")
        workflow.add_edge("viewer", "responder")
        workflow.add_edge("editor", "responder")
        workflow.add_edge("document_analyzer", "responder")
        workflow.add_edge("responder", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze query to determine intent"""
        query = state["query"]
        
        analysis_prompt = f"""
Analyze this user query and determine their intent:

USER QUERY: "{query}"

Guidelines:
- EDIT: User wants to modify, change, update, add to, alter content, rewrite, improve, optimize, or enhance documents
- SEARCH: User wants to find specific information or mentions within documents (targeted search)
- VIEW: User wants to see complete raw file content or display entire sections
- ANALYZE: User asks general questions about document content, wants summaries, or needs comprehensive understanding
- CHAT: General conversation or unclear intent

Key: Use EDIT for modifications. Use SEARCH for finding specific items. Use ANALYZE for general questions about content, summaries, or understanding documents.

Examples:
- "Find mentions of Python" → SEARCH
- "What skills do I have in machine learning?" → SEARCH  
- "What is this document about?" → ANALYZE
- "Summarize this document" → ANALYZE
- "What are the main points?" → ANALYZE
- "Show me the entire file" → VIEW
- "Rewrite this completely" → EDIT
- "Review and improve my resume" → EDIT
- "Update this document" → EDIT
- "Optimize for ATS" → EDIT

Respond with ONLY ONE WORD: EDIT, SEARCH, VIEW, ANALYZE, or CHAT
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            intent = response.content.strip().upper()
            
            intent_map = {
                "EDIT": QueryType.EDIT,
                "SEARCH": QueryType.SEARCH,
                "VIEW": QueryType.VIEW,
                "ANALYZE": QueryType.ANALYZE,
                "CHAT": QueryType.CHAT
            }
            
            state["query_type"] = intent_map.get(intent, QueryType.CHAT)
            state["confidence"] = 0.9 if intent in intent_map else 0.5
            state["entities"] = {}
            
            print(f"🧠 LLM Intent: {intent} -> {state['query_type']}")
            
        except Exception as e:
            print(f"⚠️ LLM analysis failed: {e}")
            state["query_type"] = QueryType.CHAT
            state["entities"] = {}
            state["confidence"] = 0.7
        
        return state
    
    def _route_query(self, state: AgentState) -> str:
        """Route query based on analysis"""
        query_type = state["query_type"]
        
        if query_type == QueryType.SEARCH:
            return "search"
        elif query_type == QueryType.VIEW:
            return "view"
        elif query_type == QueryType.EDIT:
            return "edit"
        elif query_type == QueryType.ANALYZE:
            return "analyze"
        else:
            return "respond"
    
    async def _search_node(self, state: AgentState) -> AgentState:
        """Execute search operation with fallback to document analysis"""
        try:
            query = state["query"]
            result = self._search_tool(query)
            
            # Check if search returned no results and fallback to document analysis
            if "No results for:" in result or "no vector index" in result.lower() or "no documents indexed" in result.lower():
                print(f"🔄 Search returned no results, falling back to document analysis")
                
                # Try to find a document to analyze
                filename = (
                    state.get("filename") or
                    self._extract_filename_from_query(query)
                )
                
                if filename:
                    file_data = await self._find_file_by_name(filename)
                    if file_data:
                        content = file_data["content"]
                        result = await self._analyze_document_content(content, query)
                    else:
                        result = f"❌ File not found: {filename}"
                else:
                    # No specific file mentioned, check if there's only one file
                    files = await self._get_workspace_files()
                    if len(files) == 1:
                        # Only one file, analyze it
                        file_data = files[0]
                        print(f"🔄 Auto-analyzing single file: {file_data['filename']}")
                        content = file_data["content"]
                        result = await self._analyze_document_content(content, query)
                    elif len(files) > 1:
                        # Multiple files, ask user to choose
                        file_list = "\n".join([f"- {f['filename']}" for f in files])
                        result = f"📄 **Multiple documents available:**\n{file_list}\n\nPlease specify which document you'd like me to search in, or I can analyze a specific one if you mention it by name."
                    else:
                        result = "📄 No documents found in this workspace."
            
            state["next_action"] = "search_completed"
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            state["error"] = f"Search failed: {str(e)}"
        return state
    
    async def _view_node(self, state: AgentState) -> AgentState:
        """Execute view operation"""
        try:
            query = state["query"]
            # Extract filename from query or use a default approach
            filename = self._extract_filename_from_query(query)
            
            if filename:
                result = await self._view_tool(filename)
            else:
                # List available files if no specific file mentioned
                files = await self._get_workspace_files()
                if files:
                    file_list = "\n".join([f"- {f['filename']}" for f in files])
                    result = f"📁 **Available files in workspace:**\n{file_list}\n\nPlease specify which file you'd like to view."
                else:
                    result = "📁 No files found in this workspace."
            
            state["next_action"] = "view_completed"
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            state["error"] = f"View failed: {str(e)}"
        return state
    
    async def _edit_node(self, state: AgentState) -> AgentState:
        """Execute edit operation"""
        try:
            query = state["query"]
            
            # Try to get filename from multiple sources (same as analyze node)
            filename = (
                state.get("filename") or  # From frontend context
                self._extract_filename_from_query(query)  # From query text
            )
            
            print(f"🔍 Debug - Edit filename from state: {state.get('filename')}")
            print(f"🔍 Debug - Edit filename from query: {self._extract_filename_from_query(query)}")
            print(f"🔍 Debug - Edit final filename: {filename}")
            
            if not filename:
                # Check if there's only one file in the workspace
                files = await self._get_workspace_files()
                if len(files) == 1:
                    filename = files[0]["filename"]
                    print(f"🔍 Debug - Auto-selecting single file for edit: {filename}")
                else:
                    file_list = "\n".join([f"- {f['filename']}" for f in files]) if files else "No files found"
                    result = f"❌ Please specify which file you want to edit.\n\n**Available files:**\n{file_list}"
                    state["next_action"] = "edit_failed"
                    state["messages"].append(AIMessage(content=result))
                    return state
            
            result = await self._llm_driven_edit(filename, query)
            state["next_action"] = "edit_completed"
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            state["error"] = f"Edit failed: {str(e)}"
        return state
    
    async def _analyze_document_node(self, state: AgentState) -> AgentState:
        """Analyze document content"""
        try:
            query = state["query"]
            
            # Try to get filename from multiple sources
            filename = (
                state.get("filename") or  # From frontend context
                self._extract_filename_from_query(query)  # From query text
            )
            
            print(f"🔍 Debug - Filename from state: {state.get('filename')}")
            print(f"🔍 Debug - Filename from query: {self._extract_filename_from_query(query)}")
            print(f"🔍 Debug - Final filename: {filename}")
            
            if filename:
                print(f"🔍 Debug - Looking for file: {filename}")
                file_data = await self._find_file_by_name(filename)
                if file_data:
                    print(f"🔍 Debug - Found file: {file_data['filename']}")
                    content = file_data["content"]
                    result = await self._analyze_document_content(content, query)
                else:
                    result = f"❌ File not found: {filename}"
            else:
                # No specific file mentioned, check if there's only one file or ask user to choose
                files = await self._get_workspace_files()
                print(f"🔍 Debug - Found {len(files)} files in workspace")
                if len(files) == 1:
                    # Only one file, analyze it
                    file_data = files[0]
                    print(f"� Debug -c Auto-analyzing single file: {file_data['filename']}")
                    content = file_data["content"]
                    result = await self._analyze_document_content(content, query)
                elif len(files) > 1:
                    # Multiple files, ask user to choose
                    file_list = "\n".join([f"- {f['filename']}" for f in files])
                    result = f"📄 **Multiple documents available:**\n{file_list}\n\nPlease specify which document you'd like me to analyze, or I can analyze a specific one if you mention it by name."
                else:
                    result = "📄 No documents found in this workspace."
            
            state["next_action"] = "analyze_completed"
            state["messages"].append(AIMessage(content=result))
        except Exception as e:
            state["error"] = f"Analysis failed: {str(e)}"
        return state
    
    def _respond_node(self, state: AgentState) -> AgentState:
        """Generate final response"""
        if state.get("error"):
            final_response = f"❌ {state['error']}"
        elif state["messages"]:
            final_response = state["messages"][-1].content
        else:
            final_response = "I'm not sure how to help with that. Could you please be more specific?"
        
        state["messages"].append(AIMessage(content=final_response))
        return state
    
    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extract filename from user query"""
        import re
        
        # Look for common file extensions
        words = query.split()
        for word in words:
            if '.' in word and any(ext in word.lower() for ext in ['.md', '.txt', '.doc', '.pdf', '.docx']):
                return word
        
        # Look for quoted filenames
        quoted_match = re.search(r'"([^"]*)"', query)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for common document references
        doc_patterns = [
            r'\b(\w+\.(?:md|txt|doc|docx|pdf))\b',
            r'\b(document|file|report|proposal)\s+(\w+)',
            r'\b(\w+)\s+(document|file|report|proposal)\b'
        ]
        
        for pattern in doc_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1) if '.' in match.group(1) else match.group(0)
        
        # Look for document type keywords that might indicate a specific file
        document_keywords = ['resume', 'cv', 'proposal', 'report', 'document', 'file']
        query_lower = query.lower()
        
        for keyword in document_keywords:
            if keyword in query_lower:
                # Try to find a file that contains this keyword in the filename
                return self._find_file_by_keyword(keyword)
        
        return None
    
    def _find_file_by_keyword(self, keyword: str) -> Optional[str]:
        """Find a file in the workspace that contains the keyword in its filename"""
        try:
            # Get all files in the workspace
            result = self.db_service.supabase.table("files").select("filename").eq("workspace_id", self.workspace_id).execute()
            
            if result.data:
                # Look for files that contain the keyword
                for file_data in result.data:
                    filename = file_data["filename"].lower()
                    if keyword.lower() in filename:
                        return file_data["filename"]
            
            return None
        except Exception as e:
            print(f"⚠️ Error finding file by keyword: {e}")
            return None
    
    async def _llm_driven_edit(self, filename: str, edit_request: str) -> str:
        """LLM-driven file editing with intelligent request handling"""
        try:
            file_data = await self._find_file_by_name(filename)
            if not file_data:
                return f"❌ File not found: {filename}"
            
            current_content = file_data["content"]
            
            # Determine the type of edit request
            edit_type = self._classify_edit_request(edit_request)
            
            if edit_type == "review_and_improve":
                # For review/improvement requests, provide analysis + improved version
                return await self._handle_review_and_improve(filename, current_content, edit_request)
            elif edit_type == "direct_edit":
                # For direct edits, modify the content directly
                return await self._handle_direct_edit(filename, current_content, edit_request)
            else:
                # Default handling
                return await self._handle_general_edit(filename, current_content, edit_request)
            
        except Exception as e:
            return f"❌ Edit error: {str(e)}"
    
    def _classify_edit_request(self, request: str) -> str:
        """Classify the type of edit request"""
        request_lower = request.lower()
        
        # Review/improvement keywords
        review_keywords = ['review', 'improve', 'highlight', 'weak areas', 'buzzwords', 'missing', 'brutally honest', 'rewrite', 'optimize', 'enhance']
        
        if any(keyword in request_lower for keyword in review_keywords):
            return "review_and_improve"
        
        # Direct edit keywords
        edit_keywords = ['add', 'remove', 'delete', 'insert', 'replace', 'update', 'change', 'modify']
        
        if any(keyword in request_lower for keyword in edit_keywords):
            return "direct_edit"
        
        return "general"
    
    def _clean_llm_response(self, content: str) -> str:
        """Remove unwanted LLM prefixes and suffixes from document content"""
        import re
        
        # Common LLM prefixes to remove
        prefixes_to_remove = [
            r"^Here is the updated document:?\s*",
            r"^Here's the updated document:?\s*",
            r"^Updated document:?\s*",
            r"^Here is the improved version:?\s*",
            r"^Here's the improved version:?\s*",
            r"^Improved version:?\s*",
            r"^Here is the rewritten document:?\s*",
            r"^Here's the rewritten document:?\s*",
            r"^Rewritten document:?\s*",
            r"^Here is the document:?\s*",
            r"^Here's the document:?\s*",
            r"^The updated document:?\s*",
            r"^The improved document:?\s*",
            r"^Document:?\s*",
        ]
        
        # Remove prefixes
        for prefix_pattern in prefixes_to_remove:
            content = re.sub(prefix_pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove common suffixes
        suffixes_to_remove = [
            r"\s*This is the updated document\.?$",
            r"\s*This is the improved version\.?$",
            r"\s*End of document\.?$",
        ]
        
        for suffix_pattern in suffixes_to_remove:
            content = re.sub(suffix_pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace
        content = content.strip()
        
        return content
    
    async def _handle_review_and_improve(self, filename: str, content: str, request: str) -> str:
        """Handle review and improvement requests"""
        
        # First, provide analysis
        analysis_prompt = f"""
You are an expert document reviewer. Analyze this document and provide detailed feedback.

DOCUMENT: {filename}
CONTENT:
{content}

USER REQUEST: "{request}"

Please provide:
1. **Weak Areas**: Identify specific weaknesses and issues
2. **Overused Buzzwords**: Point out clichéd or overused terms
3. **Missing Metrics**: Highlight where quantifiable results should be added
4. **Specific Recommendations**: Actionable improvements

Be brutally honest and specific. Focus on concrete, actionable feedback.
"""
        
        try:
            analysis_response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            analysis = analysis_response.content.strip()
            
            # Then, create improved version
            improvement_prompt = f"""
Based on the analysis, now rewrite this document to address all the identified issues.

ORIGINAL DOCUMENT:
{content}

ANALYSIS FEEDBACK:
{analysis}

Create an improved version that:
- Fixes all weak areas identified
- Replaces buzzwords with specific, impactful language
- Adds quantifiable metrics where possible
- Maintains professional tone and structure
- Is results-driven and compelling

CRITICAL: Return ONLY the improved document content. Do NOT include any prefixes like "Here is the updated document:" or explanations. Start directly with the document content.
"""
            
            improvement_response = self.llm.invoke([SystemMessage(content=improvement_prompt)])
            improved_content = self._clean_llm_response(improvement_response.content.strip())
            
            # Save the improved version
            edit_result = await self._edit_tool(filename, improved_content)
            
            # Return both analysis and confirmation
            return f"""## Document Review & Improvement Complete

### Analysis:
{analysis}

---

{edit_result}

The document has been rewritten to address all identified issues."""
            
        except Exception as e:
            return f"❌ Review and improvement error: {str(e)}"
    
    async def _handle_direct_edit(self, filename: str, content: str, request: str) -> str:
        """Handle direct edit requests"""
        edit_prompt = f"""
You are a precise document editor. Make the specific changes requested.

DOCUMENT: {filename}
CURRENT CONTENT:
{content}

USER REQUEST: "{request}"

Make the requested changes while:
- Preserving all existing content except what needs to be changed
- Maintaining the same formatting and structure
- Being precise and targeted in your edits

CRITICAL: Return ONLY the complete updated document content. Do NOT include any prefixes like "Here is the updated document:" or explanations. Start directly with the document content.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = self._clean_llm_response(response.content.strip())
            
            # Update the file in Supabase
            result = await self._edit_tool(filename, new_content)
            return result
            
        except Exception as e:
            return f"❌ Direct edit error: {str(e)}"
    
    async def _handle_general_edit(self, filename: str, content: str, request: str) -> str:
        """Handle general edit requests"""
        edit_prompt = f"""
You are a smart document editor. The user wants to edit this document.

DOCUMENT: {filename}
CURRENT CONTENT:
{content}

USER REQUEST: "{request}"

Your task:
1. Understand what the user wants to edit/add/modify
2. Find the relevant section in the document
3. Make the requested changes while preserving the document structure

Important:
- Preserve all existing content except what needs to be changed
- Maintain the same formatting and structure
- If adding content, add it in the appropriate location

CRITICAL: Return ONLY the complete updated document content. Do NOT include any prefixes like "Here is the updated document:" or explanations. Start directly with the document content.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = self._clean_llm_response(response.content.strip())
            
            # Update the file in Supabase
            result = await self._edit_tool(filename, new_content)
            return result
            
        except Exception as e:
            return f"❌ General edit error: {str(e)}"
    
    async def _analyze_document_content(self, content: str, query: str) -> str:
        """Analyze document content using LLM with intelligent context awareness"""
        
        # Detect if this is a resume analysis request
        query_lower = query.lower()
        resume_keywords = ['resume', 'cv', 'weak areas', 'buzzwords', 'metrics', 'rewrite', 'ats', 'hook', 'experience', 'format']
        
        is_resume_analysis = any(keyword in query_lower for keyword in resume_keywords)
        
        if is_resume_analysis:
            # Specialized resume analysis prompt - conversational and concise
            analysis_prompt = f"""
You are a friendly, helpful resume coach. Keep your response conversational, concise, and actionable.

RESUME CONTENT:
{content}

USER QUESTION: "{query}"

Respond in a natural, chat-like way. Be direct and helpful without being overly formal. Focus on the most important points and keep it brief. Use simple language and avoid jargon.

If they're asking for resume feedback, give 3-4 key points max. If they want specific info, just answer directly without extra fluff.
"""
        else:
            # General document analysis prompt - conversational
            analysis_prompt = f"""
You are a helpful assistant. Answer the user's question about this document in a natural, conversational way.

DOCUMENT CONTENT:
{content}

USER QUESTION: "{query}"

Keep your response:
- Conversational and friendly
- Concise and to the point  
- Easy to understand
- No unnecessary jargon or formal language

Just answer their question directly like you're having a normal conversation.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            return response.content
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    async def chat(self, message: str, filename: str = None) -> str:
        """Main chat interface"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "query": message,
                "query_type": QueryType.CHAT,
                "confidence": 0.0,
                "entities": {},
                "search_results": [],
                "file_content": None,
                "file_id": None,
                "filename": filename,
                "workspace_id": self.workspace_id,
                "user_id": self.user_id,
                "next_action": "",
                "error": None
            }
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Return the final response
            if result["messages"]:
                return result["messages"][-1].content
            else:
                return "I'm sorry, I couldn't process your request."
                
        except Exception as e:
            return f"❌ Chat error: {str(e)}"

def create_supabase_agent(workspace_id: str, user_id: str, model: str = None) -> SupabaseGraphAgent:
    """Factory function to create a Supabase-enabled agent"""
    return SupabaseGraphAgent(workspace_id, user_id, model)

    async def _handle_direct_edit(self, filename: str, content: str, request: str) -> str:
        """Handle direct edit requests with specific instructions"""
        edit_prompt = f"""You are a precise code editor. The user wants to make specific changes to this document.

DOCUMENT: {filename}
CURRENT CONTENT:
{content}

USER REQUEST: "{request}"

Your task:
1. Understand exactly what needs to be changed based on the user's request
2. Make ONLY the requested changes
3. Return the complete updated document content

Important:
- Keep all existing content except what needs to be changed
- Maintaining the same formatting and structure
- Being precise and targeted in your edits

Return ONLY the complete updated document content.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = response.content.strip()
            
            # Update the file in Supabase
            result = await self._edit_tool(filename, new_content)
            return result
            
        except Exception as e:
            return f"❌ Direct edit error: {str(e)}"
    
    async def _handle_general_edit(self, filename: str, content: str, request: str) -> str:
        """Handle general edit requests"""
        edit_prompt = f"""You are a smart document editor. The user wants to edit this document.

DOCUMENT: {filename}
CURRENT CONTENT:
{content}

USER REQUEST: "{request}"

Your task:
1. Understand what the user wants to edit/add/modify
2. Find the relevant section in the document
3. Make the requested changes while preserving the document structure
4. Return ONLY the complete updated document content

Important:
- Preserve all existing content except what needs to be changed
- Maintain the same formatting and structure
- If adding content, add it in the appropriate location
- Do not add explanations, just return the updated document
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = response.content.strip()
            
            # Update the file in Supabase
            result = await self._edit_tool(filename, new_content)
            return result
            
        except Exception as e:
            return f"❌ General edit error: {str(e)}"
    

    async def chat(self, message: str, filename: str = None) -> str:
        """Main chat interface"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "query": message,
                "query_type": QueryType.CHAT,
                "confidence": 0.0,
                "entities": {},
                "search_results": [],
                "file_content": None,
                "file_id": None,
                "filename": filename,
                "workspace_id": self.workspace_id,
                "user_id": self.user_id,
                "next_action": "",
                "error": None
            }
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Return the final response
            if result["messages"]:
                return result["messages"][-1].content
            else:
                return "I'm sorry, I couldn't process your request."
                
        except Exception as e:
            return f"❌ Agent error: {str(e)}"


def create_supabase_agent(workspace_id: str, user_id: str, model: str = None) -> SupabaseGraphAgent:
    """Factory function to create a Supabase-enabled agent"""
    return SupabaseGraphAgent(workspace_id, user_id, model)