# Vector Search Process Diagram

```mermaid
graph TD
    A[User Query: 'python decorators in lesson 3'] --> B[VectorStore.search()]
    
    B --> C[Step 1: Course Resolution]
    C --> D[Query course_catalog collection]
    D --> E[Semantic search: 'python' → 'Advanced Python Programming']
    
    B --> F[Step 2: Filter Construction]
    F --> G[Build ChromaDB filter:<br/>course='Advanced Python Programming'<br/>AND lesson_number=3]
    
    B --> H[Step 3: Content Search]
    H --> I[Embed query into 384-dim vector]
    I --> J[Search course_content collection<br/>with filter applied]
    
    J --> K[Vector Similarity Matching]
    K --> L[Calculate cosine distance<br/>between query vector and<br/>all filtered chunks]
    
    L --> M[Return top 5 results<br/>ranked by similarity]
    M --> N[SearchResults object<br/>with documents, metadata, distances]

    subgraph "ChromaDB Collections"
        O[course_catalog<br/>📚 Course Metadata]
        P[course_content<br/>📄 Chunked Content]
    end
    
    D -.-> O
    J -.-> P
    
    subgraph "Embedding Model"
        Q[SentenceTransformer<br/>all-MiniLM-L6-v2<br/>384 dimensions]
    end
    
    I -.-> Q
    
    style A fill:#e1f5fe
    style N fill:#c8e6c9
    style O fill:#fff3e0
    style P fill:#fff3e0
    style Q fill:#f3e5f5
```

## Data Flow Example

### Input Processing
```
Query: "python decorators in lesson 3"
├── Text → Vector: [0.23, -0.45, 0.12, ..., 0.78] (384 dims)
└── Parse: course_name="python", lesson_number=3
```

### Collection Structures

#### course_catalog Collection
```json
{
  "id": "Advanced_Python_Programming",
  "document": "Advanced Python Programming",
  "metadata": {
    "title": "Advanced Python Programming",
    "instructor": "Sarah Chen",
    "lesson_count": 12,
    "lessons_json": "[{\"lesson_number\": 3, \"title\": \"Decorators and Closures\"}]"
  }
}
```

#### course_content Collection
```json
{
  "id": "Advanced_Python_Programming_15",
  "document": "Decorators are a powerful feature in Python that allow you to modify or extend the behavior of functions without permanently modifying them...",
  "metadata": {
    "course_title": "Advanced Python Programming", 
    "lesson_number": 3,
    "chunk_index": 15
  }
}
```

### Similarity Scoring
```
Query Vector:     [0.23, -0.45, 0.12, ...]
Chunk 15 Vector:  [0.25, -0.43, 0.14, ...]  → Distance: 0.32 ✓
Chunk 23 Vector:  [0.87, 0.12, -0.34, ...] → Distance: 1.45
Chunk 8 Vector:   [0.21, -0.41, 0.13, ...]  → Distance: 0.28 ✓
```

### Final Results
```json
{
  "documents": [
    "Decorators are a powerful feature in Python...",
    "The @property decorator allows you to...",
    "Closure functions capture variables from..."
  ],
  "metadata": [
    {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 15},
    {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 23},
    {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 8}
  ],
  "distances": [0.28, 0.32, 0.35]
}
```