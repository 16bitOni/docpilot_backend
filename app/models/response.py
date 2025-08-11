from pydantic import BaseModel

class FileUploadResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    message: str