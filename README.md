# Document Processing API

A FastAPI-based backend service that provides intelligent document management, processing, and workspace collaboration features. Built with Supabase for data persistence, Pinecone for vector storage, and integrated AI capabilities for document analysis and editing.

## Features

- **Document Upload & Processing**: Support for PDF, DOCX, DOC, and TXT files using PyMuPDF and python-docx
- **AI-Powered Workspace Agent**: Intelligent document editing and analysis with multiple LLM models
- **Vector Embeddings**: Document search and similarity matching with Pinecone
- **Collaborative Workspaces**: Multi-user document collaboration with role-based access
- **Email Integration**: Automated notifications via Resend or Brevo SMTP
- **Real-time Chat**: Workspace-based messaging system
- **Version Control**: Automatic file versioning and change tracking

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase (PostgreSQL)
- **Vector Store**: Pinecone
- **Document Processing**: PyMuPDF, python-docx, mammoth
- **AI Models**: Groq (Llama, Mixtral, Gemma)
- **Authentication**: JWT with Supabase Auth
- **Email**: Resend API / Brevo SMTP

## Quick Start

### Prerequisites

- Python 3.8+
- Supabase account and project
- Pinecone account
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-processing-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_JWT_SECRET=your_jwt_secret
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
RESEND_API_KEY=your_resend_key  # Optional
BREVO_SMTP_USER=your_brevo_user  # Optional
BREVO_SMTP_KEY=your_brevo_key    # Optional
EMAIL_PROVIDER=auto  # auto, resend, or brevo
```

4. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

### Core Endpoints

#### Health Check
- `GET /health` - Service health status

#### Document Upload
- `POST /upload` - Upload and process documents

#### Embeddings
- `POST /embeddings` - Generate document embeddings
- `GET /embeddings/search` - Search documents by similarity

#### Workspace Agent
- `POST /api/workspace/chat` - Chat with AI agent
- `GET /api/workspace/status/{workspace_id}` - Get workspace status
- `GET /api/workspace/files/{workspace_id}` - List workspace files
- `GET /api/workspace/file/{workspace_id}/{filename}` - Get file content

### Authentication

All endpoints require JWT authentication:
```bash
curl -H "Authorization: Bearer <your-jwt-token>" \
     -X GET http://localhost:8000/api/workspace/files/workspace-id
```

### AI Models Available

- `llama3-70b-8192` - Most capable, slower response
- `llama3-8b-8192` - Faster, good for simple tasks  
- `mixtral-8x7b-32768` - Balanced speed and capability
- `gemma-7b-it` - Lightweight option

## Configuration

### Vector Store Settings (`configs/config.yaml`)
```yaml
vector_store:
  environment: "us-east-1"
  index_name: "docupilot"
  dimension: 1024

embedding:
  model: "llama-text-embed-v2"
  batch_size: 200
  chunk_size: 1000
  chunk_overlap: 200
```

### File Upload Limits
- Max file size: 10MB
- Supported formats: PDF, DOCX, DOC, TXT
- Upload directory: `uploads/`

## Database Schema

The application uses Supabase with the following main tables:

- **users** - User accounts and profiles
- **workspaces** - Document workspaces
- **files** - Document storage and metadata
- **file_versions** - Version history tracking
- **collaborators** - Workspace access control
- **chat_messages** - Workspace messaging
- **workspace_invitations** - Collaboration invites

## Usage Examples

### Upload a Document
```python
import requests

files = {'file': open('document.pdf', 'rb')}
headers = {'Authorization': 'Bearer your-jwt-token'}

response = requests.post(
    'http://localhost:8000/upload',
    files=files,
    headers=headers
)
```

### Chat with Workspace Agent
```python
import requests

data = {
    "message": "Summarize the main points in my proposal",
    "workspace_id": "workspace-uuid",
    "model": "llama3-70b-8192"
}

response = requests.post(
    'http://localhost:8000/api/workspace/chat',
    json=data,
    headers={'Authorization': 'Bearer your-jwt-token'}
)
```

### Search Documents
```python
import requests

params = {
    "query": "machine learning algorithms",
    "limit": 5
}

response = requests.get(
    'http://localhost:8000/embeddings/search',
    params=params,
    headers={'Authorization': 'Bearer your-jwt-token'}
)
```

## Development

### Project Structure
```
├── app/
│   ├── config.py          # Application settings
│   ├── dependencies.py    # FastAPI dependencies
│   ├── core/             # Core business logic
│   ├── models/           # Pydantic models
│   ├── routes/           # API endpoints
│   ├── services/         # Business services
│   └── utils/            # Utility functions
├── configs/
│   └── config.yaml       # AI and vector store config
├── uploads/              # File upload directory
├── main.py              # FastAPI application
└── requirements.txt     # Python dependencies
```

### Running Tests
```bash
pytest
```

### Code Style
```bash
black .
flake8 .
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
- Set `DEBUG=False`
- Configure proper `ALLOWED_ORIGINS`
- Use production database URLs
- Set up proper logging levels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For questions and support, please [create an issue](link-to-issues) or contact the development team.