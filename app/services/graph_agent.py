"""
Smart Graph-Based Agent using LangGraph
Analyzes queries intelligently and routes to appropriate tools
Handles modular section-based editing with state management
Now works with Supabase database instead of local files
"""

import os
import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
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
    query: str
    query_type: QueryType
    confidence: float
    entities: Dict[str, Any]
    search_results: List[Dict]
    file_content: Optional[str]
    file_path: Optional[str]
    section_start: Optional[int]
    section_end: Optional[int]
    edit_content: Optional[str]
    next_action: str
    error: Optional[str]
    context_file: Optional[str]  # File from frontend context

class SmartGraphAgent:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192",
            temperature=0.3  # Lower for more consistent analysis
        )
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index = self.pc.Index("docupilot")
        except Exception as e:
            print(f"⚠️ Pinecone initialization failed: {e}")
            self.pc = None
            self.index = None
        
        # File management
        self.base_directories = ["./temp/", "./documents/", "./data/", "./files/", os.getcwd()]
        self.file_index = {}
        self._build_file_index()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_file_index(self):
        """Build index of available files"""
        self.file_index = {}
        for base_dir in self.base_directories:
            if os.path.exists(base_dir):
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        self.file_index[file] = full_path
        print(f"📁 Indexed {len(self.file_index)} files")
    
    def _create_tools(self) -> List[StructuredTool]:
        """Create minimal, focused tools"""
        
        class SearchInput(BaseModel):
            query: str = Field(description="Search query")
            top_k: int = Field(default=5, description="Number of results")
        
        class ViewInput(BaseModel):
            filepath: str = Field(description="File path or name")
            start_line: int = Field(default=None, description="Start line (optional)")
            end_line: int = Field(default=None, description="End line (optional)")
        
        class EditInput(BaseModel):
            filepath: str = Field(description="File path or name")
            start_line: int = Field(description="Start line")
            end_line: int = Field(description="End line")
            new_content: str = Field(description="New content")
        
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
                description="View file content or sections",
                args_schema=ViewInput
            ),
            StructuredTool.from_function(
                func=self._edit_tool,
                name="edit",
                description="Edit file sections intelligently",
                args_schema=EditInput
            )
        ]
    
    def _search_tool(self, query: str, top_k: int = 5) -> str:
        """Smart search tool"""
        if not self.index:
            return "❌ Search unavailable - no vector index"
        
        try:
            # Get available namespaces
            namespaces = self._get_namespaces()
            if not namespaces:
                return "📄 No documents indexed"
            
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
    
    def _view_tool(self, filepath: str, start_line: int = None, end_line: int = None) -> str:
        """View file content tool"""
        try:
            resolved_path = self._resolve_filepath(filepath)
            if not os.path.exists(resolved_path):
                return f"❌ File not found: {filepath}"
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if start_line is None and end_line is None:
                # Show file overview
                total_lines = len(lines)
                preview = ''.join(lines[:20])
                return f"📄 **{resolved_path}** ({total_lines} lines)\n```\n{preview}\n```"
            
            start_idx = max(0, (start_line or 1) - 1)
            end_idx = min(len(lines), end_line or len(lines))
            content = ''.join(lines[start_idx:end_idx])
            
            return f"📄 **{resolved_path}** (lines {start_idx+1}-{end_idx})\n```\n{content}\n```"
            
        except Exception as e:
            return f"❌ View error: {str(e)}"
    
    def _edit_tool(self, filepath: str, start_line: int, end_line: int, new_content: str) -> str:
        """Smart section-based editing tool"""
        try:
            resolved_path = self._resolve_filepath(filepath)
            if not os.path.exists(resolved_path):
                return f"❌ File not found: {filepath}"
            
            # Read entire file
            with open(resolved_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Validate line range
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            if start_idx >= end_idx:
                return f"❌ Invalid line range: {start_line}-{end_line}"
            
            # Create backup
            backup_path = f"{resolved_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Ensure new content ends with newline
            if not new_content.endswith('\n'):
                new_content += '\n'
            
            # Replace section
            lines[start_idx:end_idx] = [new_content]
            
            # Write back
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Notify clients about file change via WebSocket
            try:
                filename = os.path.basename(resolved_path)
                tracker = get_content_tracker(os.path.dirname(resolved_path))
                diff_result = tracker.get_diff(filename)
                has_changes = diff_result.has_changes if diff_result else True
                changes_count = diff_result.changes_count if diff_result else 1
                
                # Use asyncio to run the async broadcast
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(manager.broadcast({
                        "type": "file_changed",
                        "filename": filename,
                        "has_changes": has_changes,
                        "changes": changes_count,
                        "timestamp": datetime.now().isoformat()
                    }))
                except RuntimeError:
                    # If no event loop is running, create a new one
                    asyncio.run(manager.broadcast({
                        "type": "file_changed",
                        "filename": filename,
                        "has_changes": has_changes,
                        "changes": changes_count,
                        "timestamp": datetime.now().isoformat()
                    }))
            except Exception as ws_error:
                print(f"⚠️ WebSocket notification failed: {ws_error}")
            
            return f"✅ **Edit completed**\n**File:** {resolved_path}\n**Lines:** {start_line}-{end_line}\n**Backup:** {backup_path}"
            
        except Exception as e:
            return f"❌ Edit error: {str(e)}"
    
    def _resolve_filepath(self, filepath: str) -> str:
        """Resolve file path from name or path"""
        if os.path.exists(filepath):
            return filepath
        
        if filepath in self.file_index:
            return self.file_index[filepath]
        
        # Search for partial matches
        for filename, path in self.file_index.items():
            if filepath.lower() in filename.lower():
                return path
        
        return filepath
    
    def _get_namespaces(self) -> List[str]:
        """Get available document namespaces"""
        namespaces = []
        temp_folder = "./temp/"
        if os.path.exists(temp_folder):
            for filename in os.listdir(temp_folder):
                if filename.lower().endswith(('.md', '.markdown')):
                    namespace = os.path.splitext(filename)[0].lower().replace(' ', '_').replace('-', '_')
                    namespaces.append(namespace)
        return namespaces
    
    def _filter_search_results(self, hits: List[Dict], query: str) -> List[Dict]:
        """Use LLM to filter search results for relevance"""
        if not hits:
            return hits
        
        # Prepare results for LLM evaluation
        results_text = ""
        for i, hit in enumerate(hits):
            fields = hit.get("fields", {})
            filename = fields.get("filename", "unknown")
            content = fields.get("chunk_text", "").strip()
            results_text += f"Result {i+1}: {filename}\nContent: {content}\n\n"
        
        filter_prompt = f"""
Analyze these search results for the query: "{query}"

{results_text}

Return ONLY the numbers (1, 2, 3, etc.) of results that are ACTUALLY relevant to the query "{query}".
Be strict - only include results that directly relate to what the user is asking about.

Format: Just the numbers separated by commas (e.g., "3, 4" or "2")
If no results are relevant, return "none"
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=filter_prompt)])
            relevant_indices = response.content.strip().lower()
            
            if relevant_indices == "none":
                return []
            
            # Parse the indices
            indices = []
            for idx_str in relevant_indices.split(','):
                try:
                    idx = int(idx_str.strip()) - 1  # Convert to 0-based
                    if 0 <= idx < len(hits):
                        indices.append(idx)
                except ValueError:
                    continue
            
            # Return filtered results
            return [hits[i] for i in indices] if indices else hits
            
        except Exception as e:
            print(f"⚠️ Result filtering failed: {e}")
            return hits  # Return original results if filtering fails
    
    def _answer_document_question(self, search_terms: str, original_query: str) -> str:
        """Answer 'which document' questions directly"""
        # First get search results
        search_result = self._search_tool(search_terms)
        
        # Extract document names from filtered results
        if "No relevant results found" in search_result or "No results for" in search_result:
            return f"📄 **Answer:** No documents found that discuss '{search_terms}'"
        
        # Parse the search results to extract unique document names
        lines = search_result.split('\n')
        documents = set()
        
        for line in lines:
            if '.md' in line and 'lines' in line:
                # Extract filename from lines like "**1.** filename.md (lines 1-5)"
                match = re.search(r'\*\*\d+\.\*\* (.+?\.md)', line)
                if match:
                    documents.add(match.group(1))
        
        if documents:
            doc_list = list(documents)
            if len(doc_list) == 1:
                answer = f"📄 **Answer:** The document **{doc_list[0]}** discusses '{search_terms}'"
            else:
                doc_names = "**, **".join(doc_list)
                answer = f"📄 **Answer:** These documents discuss '{search_terms}': **{doc_names}**"
            
            # Add the detailed search results
            return f"{answer}\n\n{search_result}"
        else:
            return search_result
    
    def _format_search_results(self, hits: List[Dict], query: str) -> str:
        """Format search results with intelligent filtering"""
        if not hits:
            return f"🔍 No results for: '{query}'"
        
        # Filter results using LLM
        filtered_hits = self._filter_search_results(hits, query)
        
        if not filtered_hits:
            return f"🔍 No relevant results found for: '{query}'\n\n💡 The search found some results, but they weren't directly related to your query. Try different keywords or be more specific."
        
        results = [f"🔍 **Search Results for:** '{query}' (filtered {len(filtered_hits)} relevant from {len(hits)} total)\n"]
        
        for i, hit in enumerate(filtered_hits, 1):
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
        
        # Add nodes
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
        """Let LLM analyze query intelligently without hardcoded keywords"""
        query = state["query"]
        context_file = state.get("context_file")
        
        # Create intelligent analysis prompt for LLM
        analysis_prompt = f"""
Analyze this user query and determine their intent:

USER QUERY: "{query}"
CONTEXT FILE: {context_file if context_file else "None"}

Guidelines:
- EDIT: User wants to modify, change, update, add to, or alter content
- SEARCH: User wants to find or look for information across multiple documents OR when no context file is provided
- VIEW: User wants to see raw file content or display specific sections
- ANALYZE: User asks questions about document content, wants explanations, summaries, or specific information from the document (like "what is the timeline", "tell me about", "explain", etc.)
- CHAT: General conversation or unclear intent

Important: If there's a context file and the user is asking a question about the document content (like "what is...", "tell me about...", "explain..."), choose ANALYZE, not SEARCH or VIEW.

Respond with ONLY ONE WORD: EDIT, SEARCH, VIEW, ANALYZE, or CHAT
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            intent = response.content.strip().upper()
            
            # Map LLM response to our state
            intent_map = {
                "EDIT": QueryType.EDIT,
                "SEARCH": QueryType.SEARCH,
                "VIEW": QueryType.VIEW,
                "ANALYZE": QueryType.ANALYZE,
                "CHAT": QueryType.CHAT
            }
            
            state["query_type"] = intent_map.get(intent, QueryType.CHAT)
            state["confidence"] = 0.9 if intent in intent_map else 0.5
            
            # Set entities based on intent and context
            entities = {}
            if context_file:
                entities["filename"] = context_file
            
            if state["query_type"] == QueryType.EDIT:
                entities["edit_instruction"] = query
                entities["edit_type"] = "add"  # Default, will be refined by edit analysis
            
            state["entities"] = entities
            
            print(f"🧠 LLM Intent: {intent} -> {state['query_type']}")
            
        except Exception as e:
            print(f"⚠️ LLM analysis failed: {e}")
            # Intelligent fallback based on context
            if context_file:
                # If there's a file context, try to determine intent from query
                query_lower = query.lower()
                if any(word in query_lower for word in ["update", "edit", "change", "add", "modify"]):
                    state["query_type"] = QueryType.EDIT
                    state["entities"] = {"filename": context_file, "edit_instruction": query}
                else:
                    state["query_type"] = QueryType.ANALYZE
                    state["entities"] = {"filename": context_file}
            else:
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
    
    def _search_node(self, state: AgentState) -> AgentState:
        """Execute search operation"""
        try:
            entities = state.get("entities", {})
            search_terms = entities.get("search_terms") or state["query"]
            query = state["query"]
            
            # Check if this is a "which document" question
            if any(phrase in query.lower() for phrase in ["which document", "what document", "which file", "what file"]):
                result = self._answer_document_question(search_terms, query)
            else:
                result = self._search_tool(search_terms)
            
            state["next_action"] = "search_completed"
            state["messages"].append(AIMessage(content=result))
            
        except Exception as e:
            state["error"] = f"Search failed: {str(e)}"
        
        return state
    
    def _view_node(self, state: AgentState) -> AgentState:
        """Execute view operation"""
        try:
            entities = state.get("entities", {})
            filename = entities.get("filename") or state.get("context_file")
            line_numbers = entities.get("line_numbers")
            
            if filename:
                start_line = line_numbers[0] if line_numbers else None
                end_line = line_numbers[1] if line_numbers else None
                result = self._view_tool(filename, start_line, end_line)
            else:
                result = "❌ No filename specified for viewing"
            
            state["next_action"] = "view_completed"
            state["messages"].append(AIMessage(content=result))
            
        except Exception as e:
            state["error"] = f"View failed: {str(e)}"
        
        return state
    
    def _edit_node(self, state: AgentState) -> AgentState:
        """Execute edit operation with intelligent LLM-driven approach"""
        try:
            query = state["query"]
            context_file = state.get("context_file")
            
            if not context_file:
                result = "❌ No file context provided for editing"
            else:
                # Let LLM handle the entire edit process intelligently
                result = self._llm_driven_edit(context_file, query)
            
            state["next_action"] = "edit_completed"
            state["messages"].append(AIMessage(content=result))
            
        except Exception as e:
            state["error"] = f"Edit failed: {str(e)}"
        
        return state
    
    def _llm_driven_edit(self, filename: str, edit_request: str) -> str:
        """Let LLM handle the entire edit process intelligently"""
        try:
            # Read the full file content
            resolved_path = self._resolve_filepath(filename)
            if not os.path.exists(resolved_path):
                return f"❌ File not found: {filename}"
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            # Let LLM analyze the edit request and perform the edit
            edit_prompt = f"""
You are a smart document editor. The user wants to edit this document.

DOCUMENT: {filename}
CURRENT CONTENT:
{full_content}

USER REQUEST: "{edit_request}"

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
            
            response = self.llm.invoke([SystemMessage(content=edit_prompt)])
            new_content = response.content.strip()
            
            # Create backup
            backup_path = f"{resolved_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            # Write the new content
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Notify clients about file change via WebSocket
            try:
                filename_only = os.path.basename(resolved_path)
                tracker = get_content_tracker(os.path.dirname(resolved_path))
                diff_result = tracker.get_diff(filename_only)
                has_changes = diff_result.has_changes if diff_result else True
                changes_count = diff_result.changes_count if diff_result else 1
                
                # Use asyncio to run the async broadcast
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(manager.broadcast({
                        "type": "file_changed",
                        "filename": filename_only,
                        "has_changes": has_changes,
                        "changes": changes_count,
                        "timestamp": datetime.now().isoformat()
                    }))
                except RuntimeError:
                    # If no event loop is running, create a new one
                    asyncio.run(manager.broadcast({
                        "type": "file_changed",
                        "filename": filename_only,
                        "has_changes": has_changes,
                        "changes": changes_count,
                        "timestamp": datetime.now().isoformat()
                    }))
            except Exception as ws_error:
                print(f"⚠️ WebSocket notification failed: {ws_error}")
            
            return f"✅ **Edit completed successfully!**\n\n**File:** {resolved_path}\n**Backup:** {backup_path}\n\n**Changes made:** The document has been updated according to your request."
            
        except Exception as e:
            return f"❌ Edit error: {str(e)}"
    
    def _analyze_edit_request(self, query: str, context_file: str = None) -> Dict[str, Any]:
        """Analyze edit request to extract filename, section, and edit type"""
        
        analysis = {
            "filename": context_file,  # Use context file if available
            "search_terms": None,
            "edit_instruction": query,
            "edit_type": "modify"
        }
        
        query_lower = query.lower()
        
        # If no context file, try to extract filename from query
        if not analysis["filename"]:
            if "voice ai agent development proposal" in query_lower:
                analysis["filename"] = "Voice AI Agent Development Proposal.md"
            elif "proposal" in query_lower:
                for filename in self.file_index.keys():
                    if "proposal" in filename.lower():
                        analysis["filename"] = filename
                        break
        
        # Extract section to search for
        if "core section" in query_lower:
            analysis["search_terms"] = "core system"
        elif "introduction" in query_lower:
            analysis["search_terms"] = "introduction"
        elif "experience" in query_lower:
            analysis["search_terms"] = "experience"
        else:
            analysis["search_terms"] = "core"  # Default fallback
        
        # Determine edit type
        if any(word in query_lower for word in ["add", "adding", "insert"]):
            analysis["edit_type"] = "add"
        elif any(word in query_lower for word in ["replace", "change"]):
            analysis["edit_type"] = "replace"
        elif any(word in query_lower for word in ["update", "modify", "edit"]):
            analysis["edit_type"] = "modify"
        
        return analysis
    
    def _smart_rewrite_section(self, current_section: str, edit_instruction: str, edit_type: str) -> str:
        """Intelligently rewrite a section based on edit instruction"""
        
        # Extract just the content from the view result
        content_match = re.search(r'```\n(.*?)\n```', current_section, re.DOTALL)
        if content_match:
            section_content = content_match.group(1)
        else:
            section_content = current_section
        
        rewrite_prompt = f"""
Current section content:
{section_content}

Edit instruction: {edit_instruction}
Edit type: {edit_type}

Instructions:
- If edit_type is "add": Add the new content to the existing content (usually at the end)
- If edit_type is "replace": Replace specific parts while keeping the structure
- If edit_type is "modify": Modify the content while preserving the overall structure
- If edit_type is "insert": Insert new content at appropriate location

Provide ONLY the new section content. Keep the same formatting style and structure.
Do not include explanations or markdown code blocks.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=rewrite_prompt)])
            new_content = response.content.strip()
            
            # Ensure proper formatting
            if not new_content.endswith('\n'):
                new_content += '\n'
                
            return new_content
            
        except Exception as e:
            print(f"⚠️ Section rewrite failed: {e}")
            return None
    
    def _analyze_document_node(self, state: AgentState) -> AgentState:
        """Analyze document directly without RAG for better context understanding"""
        try:
            entities = state.get("entities", {})
            filename = entities.get("filename") or state.get("context_file")
            analysis_type = entities.get("analysis_type", "summary")
            query = state["query"]
            
            if not filename:
                result = "❌ No document specified for analysis"
            else:
                # Read the full document content directly
                resolved_path = self._resolve_filepath(filename)
                if not os.path.exists(resolved_path):
                    result = f"❌ Document not found: {filename}"
                else:
                    with open(resolved_path, 'r', encoding='utf-8') as f:
                        full_content = f.read()
                    
                    # Analyze based on the specific query
                    result = self._analyze_document_content(full_content, query, filename, analysis_type)
            
            state["next_action"] = "analysis_completed"
            state["messages"].append(AIMessage(content=result))
            
        except Exception as e:
            state["error"] = f"Document analysis failed: {str(e)}"
        
        return state
    
    def _analyze_document_content(self, content: str, query: str, filename: str, analysis_type: str) -> str:
        """Analyze document content directly using LLM"""
        
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ["tell me about this", "about this document", "this document"]):
            analysis_prompt = f"""
Analyze this document and provide a comprehensive overview:

DOCUMENT: {filename}
CONTENT:
{content}

Provide a clear, structured analysis including:
1. **Document Purpose**: What is this document about?
2. **Key Sections**: What are the main sections/topics covered?
3. **Important Details**: Key information, requirements, or proposals
4. **Summary**: Brief overview of the main points

Be concise but informative.
"""
        
        elif "core system" in query_lower:
            analysis_prompt = f"""
Analyze this document and focus specifically on the core system information:

DOCUMENT: {filename}
CONTENT:
{content}

Focus on:
1. **Core System Components**: What are the main technical components?
2. **Architecture**: How is the system structured?
3. **Key Features**: What are the primary capabilities?
4. **Technical Details**: Important technical specifications

Provide a focused answer about the core system.
"""
        
        else:
            # General analysis
            analysis_prompt = f"""
Analyze this document to answer the user's question: "{query}"

DOCUMENT: {filename}
CONTENT:
{content}

Provide a direct, helpful answer to their specific question based on the document content.
If the question can't be answered from the document, say so clearly.
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            
            # Format the response nicely
            doc_name = os.path.basename(filename)
            return f"📄 **Analysis of {doc_name}**\n\n{response.content}"
            
        except Exception as e:
            return f"❌ Error analyzing document: {str(e)}"
    
    def _respond_node(self, state: AgentState) -> AgentState:
        """Generate final response"""
        if state.get("error"):
            response = f"❌ {state['error']}"
        elif state.get("next_action"):
            # Response already added by previous node
            return state
        else:
            # General chat response
            query = state["query"]
            context_file = state.get("context_file")
            
            if context_file:
                response = f"I can help you with the document **{context_file}**. You can ask me to:\n- Analyze or summarize the document\n- Search for specific information\n- Edit sections\n- View specific parts\n\nWhat would you like to know about this document?"
            else:
                response = f"I can help you search documents, edit files, and analyze content. What would you like to do?"
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    def chat(self, user_input: str, context_file: str = None) -> str:
        """Main chat interface with file context support"""
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                query=user_input,
                query_type=QueryType.CHAT,
                confidence=0.0,
                entities={},
                search_results=[],
                file_content=None,
                file_path=None,
                section_start=None,
                section_end=None,
                edit_content=None,
                next_action="",
                error=None,
                context_file=context_file
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Return the last AI message
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            return ai_messages[-1].content if ai_messages else "❌ No response generated"
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

def create_graph_agent() -> SmartGraphAgent:
    """Factory function to create the graph agent"""
    return SmartGraphAgent()

# Example usage
if __name__ == "__main__":
    agent = create_graph_agent()
    
    print("🤖 Smart Graph Agent initialized!")
    print("Try: 'search for AI policy', 'edit the introduction section', 'show me document.md'")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        
        response = agent.chat(user_input)
        print(f"\n🤖 {response}")