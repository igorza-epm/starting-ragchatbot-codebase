import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    def validate_api_key(self) -> None:
        """
        Validate that the Anthropic API key is present and properly formatted.
        Raises ValueError if the API key is invalid.
        """
        if not self.ANTHROPIC_API_KEY:
            error_msg = (
                "\n❌ ERROR: Anthropic API key not found!\n"
                "Please set the ANTHROPIC_API_KEY in your .env file.\n"
                "Example: ANTHROPIC_API_KEY=sk-ant-api03-...\n"
                "Get your API key from: https://console.anthropic.com/settings/keys"
            )
            raise ValueError(error_msg)

        # Check for proper format (Anthropic keys start with 'sk-ant-')
        if not self.ANTHROPIC_API_KEY.startswith('sk-ant-'):
            error_msg = (
                "\n⚠️  WARNING: API key format appears incorrect.\n"
                "Anthropic API keys should start with 'sk-ant-'.\n"
                "Please verify your API key is correct."
            )
            print(error_msg, file=sys.stderr)

        # Check for placeholder or test values
        if 'your' in self.ANTHROPIC_API_KEY.lower() or self.ANTHROPIC_API_KEY == 'test-api-key':
            error_msg = (
                "\n❌ ERROR: Invalid API key detected!\n"
                "Please replace the placeholder with your actual Anthropic API key.\n"
                "Get your API key from: https://console.anthropic.com/settings/keys"
            )
            raise ValueError(error_msg)

config = Config()


