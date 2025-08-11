import logging
from typing import Dict, Any
from datetime import datetime
import uuid

from supabase import create_client, Client
from app.config import settings
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

class DatabaseService:
    """Database service for saving converted files to Supabase"""
    
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    async def get_or_create_default_workspace(self, user_id: str) -> str:
        """Get or create a default workspace for the user"""
        try:
            # Try to get existing workspace owned by user
            result = self.supabase.table("workspaces").select("id").eq("owner_id", user_id).limit(1).execute()
            
            if result.data:
                return result.data[0]["id"]
            
            # Create a default workspace if none exists
            workspace_data = {
                "name": "Default Workspace",
                "owner_id": user_id,
                "description": "Auto-created workspace for file uploads"
            }
            
            workspace_result = self.supabase.table("workspaces").insert(workspace_data).execute()
            
            if workspace_result.data:
                return workspace_result.data[0]["id"]
            else:
                # If workspace creation fails, return None to skip workspace_id
                return None
                
        except Exception as e:
            logger.warning(f"Could not create workspace for user {user_id}: {str(e)}")
            return None

    async def save_converted_file(
        self,
        filename: str,
        file_type: str,
        content: str,
        user_id: str,
        workspace_id: str = None
    ) -> Dict[str, Any]:
        """Save converted markdown file to Supabase"""
        try:
            # Get or create a workspace if none provided
            if not workspace_id:
                workspace_id = await self.get_or_create_default_workspace(user_id)
            
            # Ensure we have a workspace_id (required by schema)
            if not workspace_id:
                raise AppException("Could not create or find workspace for user", 500)
            
            file_data = {
                "filename": filename,
                "file_type": file_type,
                "content": content,
                "created_by": user_id,
                "workspace_id": workspace_id
            }
            
            # Insert into Supabase
            result = self.supabase.table("files").insert(file_data).execute()
            
            if result.data:
                logger.info(f"Saved converted file {filename} for user {user_id}")
                return result.data[0]
            else:
                raise AppException("Failed to save file to database", 500)
                
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            raise AppException("Database operation failed", 500, str(e))

    async def get_user_files(self, user_id: str) -> list:
        """Get all converted files for a user"""
        try:
            result = self.supabase.table("files").select("*").eq("created_by", user_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get user files: {str(e)}")
            raise AppException("Failed to retrieve files", 500, str(e))

    async def get_file_by_id(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """Get a specific file by ID (with user verification)"""
        try:
            result = self.supabase.table("files").select("*").eq("id", file_id).eq("created_by", user_id).execute()
            
            if result.data:
                return result.data[0]
            else:
                raise AppException("File not found", 404)
                
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Failed to get file: {str(e)}")
            raise AppException("Failed to retrieve file", 500, str(e))   
            
    async def get_workspace_files(self, workspace_id: str, user_id: str) -> list:
        """Get all files in a specific workspace (with user verification)"""
        try:
            print(f"🔍 DB Debug - Getting files for user {user_id} in workspace {workspace_id}")
            
            # Check if user is in the collaborators table (includes owner, editor, viewer)
            collab_result = self.supabase.table("collaborators").select("role").eq("workspace_id", workspace_id).eq("user_id", user_id).execute()
            
            print(f"🔍 DB Debug - Collaborator check: {collab_result.data}")
            
            has_access = False
            
            if collab_result.data:
                has_access = True
                user_role = collab_result.data[0]['role']
                print(f"✅ DB Debug - User found with role: {user_role}")
            else:
                # Fallback: Check if user is the workspace owner (in case they're not in collaborators table)
                workspace_result = self.supabase.table("workspaces").select("id").eq("id", workspace_id).eq("owner_id", user_id).execute()
                
                print(f"🔍 DB Debug - Fallback owner check: {workspace_result.data}")
                
                if workspace_result.data:
                    has_access = True
                    print(f"✅ DB Debug - User is workspace owner (fallback)")
            
            if not has_access:
                print(f"❌ DB Debug - No access found")
                raise AppException("Workspace not found or access denied", 404)
            
            # Get files in the workspace
            result = self.supabase.table("files").select("*").eq("workspace_id", workspace_id).execute()
            print(f"🔍 DB Debug - Files found: {len(result.data) if result.data else 0}")
            return result.data if result.data else []
            
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Failed to get workspace files: {str(e)}")
            raise AppException("Failed to retrieve workspace files", 500, str(e))