# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start using shell script
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new package
uv add package_name
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Python 3.13+ required
- Uses `uv` as package manager

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials with a modular, tool-based architecture:

### Core Components

**RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- Manages document processing, vector storage, AI generation, and session management
- Uses tool-based search pattern with ToolManager for extensible functionality

**VectorStore** (`backend/vector_store.py`): ChromaDB-based storage with dual collections
- `course_catalog`: Stores course metadata for semantic course discovery
- `course_content`: Stores chunked course content for retrieval
- Supports filtered search by course name and lesson number

**AIGenerator** (`backend/ai_generator.py`): Claude API integration with tool support
- Uses tool-calling pattern for structured search operations
- Handles conversation history and multi-turn interactions
- Static system prompt optimized for educational content

**DocumentProcessor** (`backend/document_processor.py`): Converts documents into structured Course/Lesson models
- Chunks text content for optimal vector storage
- Extracts course structure from document content

### Key Design Patterns

1. **Tool-Based Architecture**: AI uses search tools rather than direct retrieval, enabling more sophisticated query handling
2. **Dual Collection Storage**: Separates course metadata from content for efficient filtering and search
3. **Session Management**: Tracks conversation history for contextual responses
4. **Modular Components**: Each component has single responsibility with clear interfaces

### Data Models (`backend/models.py`)
- **Course**: Represents complete course with lessons
- **Lesson**: Individual lesson within a course
- **CourseChunk**: Text chunk for vector storage with metadata

### API Structure
- FastAPI backend serving both API endpoints and static frontend
- `/api/query`: Main query endpoint with session support
- `/api/courses`: Course analytics and metadata
- Auto-loads documents from `docs/` folder on startup

### Frontend Integration
- Simple HTML/CSS/JavaScript frontend in `frontend/`
- Backend serves static files and API from same port (8000)
- Real-time interaction with session persistence

## File Processing
- Supports PDF, DOCX, and TXT files in `docs/` folder
- Automatic chunking with configurable size (800 chars) and overlap (100 chars)
- Prevents duplicate course processing based on course titles
- Use uv to run python files