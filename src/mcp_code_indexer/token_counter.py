"""
Token counting functionality using tiktoken with offline cache.

This module provides token counting capabilities using the tiktoken library
with a bundled cache file for offline operation. It enables accurate token
estimation for determining whether to use full overview or search approaches.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import tiktoken

from mcp_code_indexer.database.models import FileDescription, FolderNode

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Handles token counting using tiktoken with offline cache support.

    Automatically configures tiktoken to use bundled cache file for offline
    operation and provides methods to count tokens in various data structures.
    """

    def __init__(self, token_limit: int = 32000):
        """
        Initialize token counter with specified limit.

        Args:
            token_limit: Maximum tokens before recommending search over overview
        """
        self.token_limit = token_limit
        self._encoder: Optional[tiktoken.Encoding] = None
        self._setup_offline_tiktoken()
        self._init_encoder()

    def _setup_offline_tiktoken(self) -> None:
        """Configure tiktoken to use bundled encoding file for offline operation."""
        # Get path to bundled cache directory
        base_dir = Path(__file__).parent.absolute()
        cache_dir = base_dir / "tiktoken_cache"

        # Ensure cache directory exists
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Tiktoken cache directory not found at {cache_dir}. "
                "Please ensure the tiktoken_cache directory exists in the src folder."
            )

        # Set tiktoken to use our bundled cache
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)

        # Verify the encoding file exists
        cache_file = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
        cache_path = cache_dir / cache_file

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Tiktoken cache file not found at {cache_path}. "
                "Please ensure the cl100k_base.tiktoken file is properly "
                f"renamed to {cache_file} and placed in the tiktoken_cache directory."
            )

        logger.debug(f"Configured tiktoken to use cache at {cache_dir}")

    def _init_encoder(self) -> None:
        """Initialize tiktoken encoder with fallback options."""
        try:
            # Try to get the cl100k_base encoding directly
            self._encoder = tiktoken.get_encoding("cl100k_base")
            logger.debug("Initialized tiktoken with cl100k_base encoding")
        except Exception as e:
            logger.warning(f"Failed to load cl100k_base encoding: {e}")
            try:
                # Fallback to model-based encoding
                self._encoder = tiktoken.encoding_for_model("gpt-4o")
                logger.debug("Initialized tiktoken with gpt-4o model encoding")
            except Exception as fallback_error:
                raise RuntimeError(
                    "Failed to initialize tiktoken encoder. "
                    "Check that the cache file is properly configured and accessible."
                ) from fallback_error

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Get the tiktoken encoder instance."""
        if self._encoder is None:
            raise RuntimeError("Token encoder not properly initialized")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Input text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0

        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Failed to count tokens for text: {e}")
            # Fallback to rough approximation (4 chars per token)
            return len(text) // 4

    def count_file_description_tokens(self, file_desc: FileDescription) -> int:
        """
        Count tokens for a file description in overview format.

        Args:
            file_desc: File description to count tokens for

        Returns:
            Number of tokens for formatted file description
        """
        # Format matches what would be shown in codebase overview
        formatted_content = f"{file_desc.file_path}\n{file_desc.description}\n"
        return self.count_tokens(formatted_content)

    def count_folder_structure_tokens(self, folder: FolderNode) -> int:
        """
        Count tokens for a complete folder structure.

        Args:
            folder: Root folder node to count tokens for

        Returns:
            Total number of tokens for the folder structure
        """
        total_tokens = 0

        # Count tokens for folder name and path
        folder_header = f"{folder.name}/\n"
        total_tokens += self.count_tokens(folder_header)

        # Count tokens for all files in this folder
        for file_node in folder.files:
            file_content = f"{file_node.path}\n{file_node.description}\n"
            total_tokens += self.count_tokens(file_content)

        # Recursively count tokens for subfolders
        for subfolder in folder.folders:
            total_tokens += self.count_folder_structure_tokens(subfolder)

        return total_tokens

    def calculate_codebase_tokens(
        self, file_descriptions: List[FileDescription]
    ) -> int:
        """
        Calculate total tokens for a list of file descriptions.

        Args:
            file_descriptions: List of file descriptions to count

        Returns:
            Total token count for all file descriptions
        """
        total_tokens = 0

        for file_desc in file_descriptions:
            total_tokens += self.count_file_description_tokens(file_desc)

        return total_tokens

    def is_large_codebase(self, total_tokens: int) -> bool:
        """
        Check if codebase exceeds configured token limit.

        Args:
            total_tokens: Total token count to check

        Returns:
            True if codebase exceeds token limit
        """
        return total_tokens > self.token_limit

    def get_recommendation(self, total_tokens: int) -> str:
        """
        Get recommendation for codebase navigation approach.

        Args:
            total_tokens: Total token count

        Returns:
            "use_search" or "use_overview" based on token count
        """
        return "use_search" if self.is_large_codebase(total_tokens) else "use_overview"

    def generate_cache_key(
        self, project_id: str, branch: str, content_hash: str
    ) -> str:
        """
        Generate a cache key for token count caching.

        Args:
            project_id: Project identifier
            branch: Git branch name
            content_hash: Hash of file contents or descriptions

        Returns:
            Cache key string
        """
        key_content = f"{project_id}:{branch}:{content_hash}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:16]


def verify_tiktoken_setup() -> bool:
    """
    Verify that tiktoken is properly configured for offline operation.

    Returns:
        True if tiktoken setup is working correctly
    """
    try:
        counter = TokenCounter()

        # Test with a known string
        test_string = "Hello, world!"
        token_count = counter.count_tokens(test_string)

        # cl100k_base should encode "Hello, world!" to 4 tokens
        expected_count = 4

        if token_count == expected_count:
            logger.info("Tiktoken offline setup verified successfully")
            return True
        else:
            logger.warning(
                (
                    f"Tiktoken token count mismatch: expected {expected_count}, "
                    f"got {token_count}"
                )
            )
            return False

    except Exception as e:
        logger.error(f"Tiktoken setup verification failed: {e}")
        return False
