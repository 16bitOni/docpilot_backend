"""
API router for agent interactions - Graph-based smart agent.
Updated to support both local files (legacy) and Supabase workspace integration.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

from app.services.graph_agent import SmartGraphAgent
from app.services.supabase_graph_agent import create_supabase_agent
from app.dependencies import get_current_user
from app.services.database_service import DatabaseService
from app.services.auth_service import AuthService
from app.utils.exceptions import AppException

security = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/agent", tags=["agent"])

# Legacy agent instance for backward compatibility
legacy_agent_instance: Optional[SmartGraphAgent] = None


class ChatRequest(BaseModel):
    message: str
    clear_history: bool = False
    context_file: str = None  # File from frontend context
    workspace_id: Optional[str] = None  # New: workspace context for Supabase mode


class ChatResponse(BaseModel):
    response: str
    mode: str = "legacy"  # "legacy" or "workspace"




def get_legacy_agent() -> SmartGraphAgent:
    """Get or create legacy agent instance for backward compatibility."""
    global legacy_agent_instance
    if legacy_agent_instance is None:
        # Import here to avoid circular imports
        from app.services.graph_agent import SmartGraphAgent
        legacy_agent_instance = SmartGraphAgent()
    return legacy_agent_instance


async def get_optional_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    """Get current user if token is provided, otherwise return None."""
    if not credentials:
        return None
    
    try:
        auth_service = AuthService()
        user = await auth_service.validate_token(credentials.credentials)
        return user
    except Exception:
        return None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest, current_user: Optional[dict] = Depends(get_optional_current_user)):
    """
    Chat with the smart assistant.
    Supports both legacy mode (local files) and workspace mode (Supabase).
    """
    try:
        # Determine mode based on workspace_id presence
        if request.workspace_id:
            # Workspace mode - use Supabase agent (requires authentication)
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required for workspace mode")
            
            user_id = current_user["id"]
            
            # Verify workspace access
            db_service = DatabaseService()
            await _verify_workspace_access(db_service, request.workspace_id, user_id)
            
            # Create workspace-scoped agent
            agent = create_supabase_agent(request.workspace_id, user_id)
            response = await agent.chat(request.message, request.context_file)
            
            return ChatResponse(response=response, mode="workspace")
        else:
            # Legacy mode - use local file agent (authentication optional)
            agent = get_legacy_agent()
            response = agent.chat(request.message, context_file=request.context_file)
            
            return ChatResponse(response=response, mode="legacy")
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.get("/status")
async def get_agent_status(workspace_id: Optional[str] = None, current_user: Optional[dict] = Depends(get_optional_current_user)):
    """Get agent status and conversation summary."""
    try:
        if workspace_id:
            # Workspace mode status (requires authentication)
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required for workspace mode")
            
            user_id = current_user["id"]
            db_service = DatabaseService()
            await _verify_workspace_access(db_service, workspace_id, user_id)
            
            files = await db_service.get_workspace_files(workspace_id, user_id)
            
            return {
                "status": "active",
                "agent_type": "supabase_graph_based",
                "mode": "workspace",
                "workspace_id": workspace_id,
                "available_tools": ["search", "view", "edit"],
                "available_files": len(files),
                "user_id": user_id
            }
        else:
            # Legacy mode status (authentication optional)
            agent = get_legacy_agent()
            
            return {
                "status": "active",
                "agent_type": "graph_based",
                "mode": "legacy",
                "available_tools": ["search", "view", "edit"],
                "indexed_files": len(agent.file_index),
                "available_namespaces": len(agent._get_namespaces())
            }
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/clear")
async def clear_conversation():
    """Clear the conversation history."""
    try:
        # Graph agent is stateless, so clearing means creating a new instance
        global legacy_agent_instance
        legacy_agent_instance = None
        
        return {"message": "Agent state cleared - new instance will be created on next request"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


@router.get("/conversation-summary")
async def get_conversation_summary():
    """
    Get conversation summary.
    """
    try:
        # Graph agent is stateless, so no persistent conversation history
        return {
            "conversation_summary": "Graph agent is stateless - each request is independent",
            "message_count": 0,
            "note": "Use chat endpoint for individual interactions"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation summary error: {str(e)}")


async def _verify_workspace_access(db_service: DatabaseService, workspace_id: str, user_id: str):
    """Verify that the user has access to the workspace."""
    try:
        # Check if user is the owner
        owner_result = db_service.supabase.table("workspaces").select("id").eq("id", workspace_id).eq("owner_id", user_id).execute()
        
        if owner_result.data:
            return True
        
        # Check if user is a collaborator
        collab_result = db_service.supabase.table("collaborators").select("id").eq("workspace_id", workspace_id).eq("user_id", user_id).execute()
        
        if collab_result.data:
            return True
        
        # No access found
        raise AppException("Workspace not found or access denied", 403)
        
    except AppException:
        raise
    except Exception as e:
        raise AppException("Failed to verify workspace access", 500, str(e))

