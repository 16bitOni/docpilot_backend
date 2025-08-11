# Workspace Agent API Documentation

The Workspace Agent API provides intelligent document management and editing capabilities within workspace contexts using Supabase database integration.

## Authentication

All endpoints require JWT authentication via Bearer token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### 1. Chat with Workspace Agent

**POST** `/api/workspace/chat`

Chat with the intelligent agent in a specific workspace context.

**Request Body:**
```json
{
  "message": "Edit the introduction section of my proposal",
  "workspace_id": "uuid-of-workspace",
  "model": "llama3-70b-8192",  // optional - AI model selection
  "filename": "proposal.md"    // optional - specific file context
}
```

**Available AI Models:**
- `llama3-70b-8192` (default) - Most capable, slower
- `llama3-8b-8192` - Faster, good for simple tasks
- `mixtral-8x7b-32768` - Good balance of speed and capability
- `gemma-7b-it` - Lightweight option

**Response:**
```json
{
  "response": "✅ Edit completed successfully! The document has been updated in the database.",
  "workspace_id": "uuid-of-workspace"
}
```

**Example Usage:**

```bash
curl -X POST "http://localhost:8000/api/workspace/chat" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me the content of my project proposal",
    "workspace_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

### 2. Get Workspace Agent Status

**GET** `/api/workspace/status/{workspace_id}`

Get the status and information about the agent for a specific workspace.

**Response:**
```json
{
  "status": "active",
  "workspace_id": "uuid-of-workspace",
  "user_id": "uuid-of-user",
  "available_files": 5,
  "agent_type": "supabase_graph_based"
}
```

### 3. Get Workspace Files

**GET** `/api/workspace/files/{workspace_id}`

Get all files available in the workspace for agent operations.

**Response:**
```json
{
  "success": true,
  "workspace_id": "uuid-of-workspace",
  "files": [
    {
      "id": "file-uuid",
      "filename": "proposal.md",
      "file_type": ".md",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

### 4. Get Specific File Content

**GET** `/api/workspace/file/{workspace_id}/{filename}`

Get the content of a specific file in the workspace.

**Response:**
```json
{
  "success": true,
  "file": {
    "id": "file-uuid",
    "filename": "proposal.md",
    "content": "# My Proposal\n\nThis is the content...",
    "file_type": ".md",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

## Agent Capabilities

The workspace agent can understand and execute various types of requests:

### 1. **View Operations**
- "Show me the content of proposal.md"
- "Display the introduction section"
- "What files are available in this workspace?"

### 2. **Edit Operations**
- "Update the timeline in my project plan"
- "Add a new section about budget to the proposal"
- "Change the conclusion paragraph"

### 3. **Search Operations**
- "Find documents that mention 'machine learning'"
- "Search for information about project timelines"

### 4. **Analysis Operations**
- "What is the main topic of this document?"
- "Summarize the key points in my proposal"
- "Explain the project timeline"

## Agent Features

### 🔍 **Intelligent Query Analysis**
The agent uses LLM to understand user intent and route requests to appropriate tools.

### 📁 **Supabase Integration**
- Files are stored and managed in Supabase database
- Real-time updates and version control
- Workspace-based access control

### 🔐 **Security**
- JWT-based authentication
- Workspace access verification
- User permission checking

### 📝 **Smart Editing**
- LLM-driven content editing
- Automatic backup creation (file versions)
- Structure-preserving modifications

### 🔄 **Real-time Updates**
- WebSocket notifications for file changes
- Live collaboration support

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `401` - Unauthorized (invalid JWT)
- `403` - Forbidden (no workspace access)
- `404` - Not found (workspace/file not found)
- `500` - Internal server error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Integration Example

Here's a complete example of integrating the workspace agent into a frontend application:

```javascript
class WorkspaceAgent {
  constructor(apiUrl, jwtToken) {
    this.apiUrl = apiUrl;
    this.token = jwtToken;
  }

  async chat(workspaceId, message, filename = null) {
    const response = await fetch(`${this.apiUrl}/api/workspace/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        workspace_id: workspaceId,
        filename
      })
    });

    if (!response.ok) {
      throw new Error(`Agent error: ${response.statusText}`);
    }

    return await response.json();
  }

  async getWorkspaceFiles(workspaceId) {
    const response = await fetch(`${this.apiUrl}/api/workspace/files/${workspaceId}`, {
      headers: {
        'Authorization': `Bearer ${this.token}`
      }
    });

    return await response.json();
  }
}

// Usage
const agent = new WorkspaceAgent('http://localhost:8000', 'your-jwt-token');

// Chat with the agent
const result = await agent.chat(
  'workspace-uuid', 
  'Edit the introduction section of my proposal'
);

console.log(result.response);
```

## Database Schema Integration

The agent works with these Supabase tables:

- **workspaces** - Workspace information and ownership
- **files** - File content and metadata
- **file_versions** - Version history and backups
- **collaborators** - Workspace access control
- **users** - User information

The agent automatically handles:
- Workspace access verification
- File version creation on edits
- Real-time change notifications
- User permission checking