"""
API router for workspace-based agent interactions with Supabase integration.
Handles authentication and workspace-scoped file operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from app.dependencies import get_current_user
from app.services.supabase_graph_agent_v2 import create_supabase_agent
from app.services.database_service import DatabaseService
from app.utils.exceptions import AppException

router = APIRouter(prefix="/workspace", tags=["workspace-agent"])

class WorkspaceChatRequest(BaseModel):
    message: str
    workspace_id: str
    model: Optional[str] = None  # AI model selection (e.g., "llama3-70b-8192", "llama3-8b-8192")
    filename: Optional[str] = None  # Optional specific file context

class WorkspaceChatResponse(BaseModel):
    response: str
    workspace_id: str

class WorkspaceStatusResponse(BaseModel):
    status: str
    workspace_id: str
    user_id: str
    available_files: int
    agent_type: str

@router.post("/chat", response_model=WorkspaceChatResponse)
async def chat_with_workspace_agent(
    request: WorkspaceChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Chat with the smart assistant in a specific workspace context.
    Requires JWT authentication and workspace access verification.
    """
    try:
        user_id = current_user["id"]
        workspace_id = request.workspace_id
        
        # Verify user has access to the workspace (viewers can chat)
        db_service = DatabaseService()
        access_info = await _verify_workspace_access(db_service, workspace_id, user_id, "viewer")
        
        # Create workspace-scoped agent with optional model selection
        agent = create_supabase_agent(workspace_id, user_id, request.model)
        
        # Check if this is an edit request and verify editor permissions
        message_lower = request.message.lower()
        edit_keywords = ['edit', 'update', 'change', 'modify', 'rewrite', 'improve', 'add', 'remove', 'delete', 'review', 'fix']
        
        is_edit_request = any(keyword in message_lower for keyword in edit_keywords)
        
        if is_edit_request and access_info["role"] not in ["owner", "editor"]:
            raise HTTPException(
                status_code=403, 
                detail="Edit operations require editor or owner permissions"
            )
        
        # Get response from agent
        response = await agent.chat(request.message, request.filename)
        
        return WorkspaceChatResponse(
            response=response,
            workspace_id=workspace_id
        )
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@router.get("/status/{workspace_id}", response_model=WorkspaceStatusResponse)
async def get_workspace_agent_status(
    workspace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get agent status for a specific workspace."""
    try:
        user_id = current_user["id"]
        
        # Verify workspace access (viewers can check status)
        db_service = DatabaseService()
        access_info = await _verify_workspace_access(db_service, workspace_id, user_id, "viewer")
        
        # Get workspace files count
        files = await db_service.get_workspace_files(workspace_id, user_id)
        
        return WorkspaceStatusResponse(
            status="active",
            workspace_id=workspace_id,
            user_id=user_id,
            available_files=len(files),
            agent_type="supabase_graph_based"
        )
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@router.get("/files/{workspace_id}")
async def get_workspace_files(
    workspace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all files in a workspace for the agent context."""
    try:
        user_id = current_user["id"]
        
        # Verify workspace access (viewers can see files)
        db_service = DatabaseService()
        access_info = await _verify_workspace_access(db_service, workspace_id, user_id, "viewer")
        
        # Get workspace files
        files = await db_service.get_workspace_files(workspace_id, user_id)
        
        return {
            "success": True,
            "workspace_id": workspace_id,
            "files": [
                {
                    "id": f["id"],
                    "filename": f["filename"],
                    "file_type": f["file_type"],
                    "created_at": f["created_at"],
                    "updated_at": f["updated_at"]
                }
                for f in files
            ],
            "count": len(files)
        }
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workspace files: {str(e)}")

@router.get("/file/{workspace_id}/{filename}")
async def get_workspace_file_content(
    workspace_id: str,
    filename: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific file content from workspace."""
    try:
        user_id = current_user["id"]
        
        # Verify workspace access (viewers can read file content)
        db_service = DatabaseService()
        access_info = await _verify_workspace_access(db_service, workspace_id, user_id, "viewer")
        
        # Create agent to use its file finding logic
        agent = create_supabase_agent(workspace_id, user_id)
        file_data = await agent._find_file(filename)
        
        if not file_data:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        return {
            "success": True,
            "file": {
                "id": file_data["id"],
                "filename": file_data["filename"],
                "content": file_data["content"],
                "file_type": file_data["file_type"],
                "created_at": file_data["created_at"],
                "updated_at": file_data["updated_at"]
            }
        }
    
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file: {str(e)}")

async def _verify_workspace_access(db_service: DatabaseService, workspace_id: str, user_id: str, required_role: str = "viewer"):
    """Verify that the user has access to the workspace with the required role."""
    try:
        print(f"🔍 Debug - Checking access for user {user_id} to workspace {workspace_id}")
        
        # Check if user is in the collaborators table (which includes owner, editor, viewer)
        collab_result = db_service.supabase.table("collaborators").select("role").eq("workspace_id", workspace_id).eq("user_id", user_id).execute()
        
        print(f"🔍 Debug - Collaborator check result: {collab_result.data}")
        
        if collab_result.data:
            user_role = collab_result.data[0]["role"]
            print(f"✅ User found with role: {user_role}")
            
            # Define role hierarchy: owner > editor > viewer
            role_hierarchy = {"owner": 3, "editor": 2, "viewer": 1}
            
            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(required_role, 1)
            
            if user_level >= required_level:
                return {"access": True, "role": user_role}
            else:
                raise AppException(f"Insufficient permissions. Required: {required_role}, User has: {user_role}", 403)
        
        # Fallback: Check if user is the workspace owner (in case they're not in collaborators table)
        owner_result = db_service.supabase.table("workspaces").select("id").eq("id", workspace_id).eq("owner_id", user_id).execute()
        
        print(f"🔍 Debug - Fallback owner check result: {owner_result.data}")
        
        if owner_result.data:
            print(f"✅ User is workspace owner (fallback)")
            return {"access": True, "role": "owner"}
        
        # No access found
        print(f"❌ No access found - user not found in collaborators or as owner")
        raise AppException("Workspace not found or access denied", 403)
        
    except AppException:
        raise
    except Exception as e:
        print(f"❌ Access verification error: {str(e)}")
        raise AppException("Failed to verify workspace access", 500, str(e))