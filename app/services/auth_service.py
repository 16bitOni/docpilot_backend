import jwt
from typing import Dict, Any
import logging

from app.config import settings
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        self.jwt_secret = settings.SUPABASE_JWT_SECRET

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return user info"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=["HS256"],
                options={"verify_aud": False}  # Supabase tokens don't always have aud
            )
            
            # Extract user info
            user_info = {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "role": payload.get("role", "authenticated")
            }
            
            if not user_info["id"]:
                raise AppException("Invalid token: missing user ID", 401)
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            raise AppException("Token has expired", 401)
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise AppException("Invalid token", 401)
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            raise AppException("Authentication failed", 401)