#!/usr/bin/env python3
"""
Base Claude API Handler for MCP Code Indexer

Provides shared functionality for interacting with Claude via OpenRouter API,
including token management, retry logic, and response validation.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

import aiohttp
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from .database.database import DatabaseManager
from .token_counter import TokenCounter


class ClaudeAPIError(Exception):
    """Base exception for Claude API operations."""

    pass


class ClaudeRateLimitError(ClaudeAPIError):
    """Exception for rate limiting scenarios."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class ClaudeValidationError(ClaudeAPIError):
    """Exception for response validation failures."""

    pass


@dataclass
class ClaudeConfig:
    """Configuration for Claude API calls."""

    model: str = "anthropic/claude-sonnet-4"
    max_tokens: int = 24000
    temperature: float = 0.3
    timeout: int = 300
    token_limit: int = 180000


@dataclass
class ClaudeResponse:
    """Structured response from Claude API."""

    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class ClaudeAPIHandler:
    """
    Base handler for Claude API interactions via OpenRouter.

    Provides shared functionality for:
    - Token counting and limit validation
    - API request/response handling with retry logic
    - Response validation and parsing
    - Error handling and logging
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Claude API Handler.

        Args:
            db_manager: Database manager instance
            cache_dir: Cache directory for temporary files
            logger: Logger instance to use (optional, creates default if not provided)
        """
        self.db_manager = db_manager
        self.cache_dir = cache_dir
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.token_counter = TokenCounter()

        # Initialize configuration
        self.config = ClaudeConfig(
            model=os.getenv("MCP_CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
            max_tokens=int(os.getenv("MCP_CLAUDE_MAX_TOKENS", "24000")),
            temperature=float(os.getenv("MCP_CLAUDE_TEMPERATURE", "0.3")),
            timeout=int(os.getenv("MCP_CLAUDE_TIMEOUT", "600")),  # 10 minutes
            token_limit=int(os.getenv("MCP_CLAUDE_TOKEN_LIMIT", "180000")),
        )

        # Validate API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ClaudeAPIError("OPENROUTER_API_KEY environment variable is required")

    def validate_token_limit(self, prompt: str, context: str = "") -> bool:
        """
        Validate that prompt + context fits within token limit.

        Args:
            prompt: Main prompt text
            context: Additional context (project overview, file descriptions, etc.)

        Returns:
            True if within limits, False otherwise
        """
        combined_text = f"{prompt}\n\n{context}"
        token_count = self.token_counter.count_tokens(combined_text)

        self.logger.debug(
            f"Token count validation: {token_count}/{self.config.token_limit}"
        )

        if token_count > self.config.token_limit:
            self.logger.warning(
                f"Token limit exceeded: {token_count} > {self.config.token_limit}. "
                f"Consider using shorter context or ask for a more specific question."
            )
            return False

        return True

    def get_token_count(self, text: str) -> int:
        """Get token count for given text."""
        return self.token_counter.count_tokens(text)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ClaudeRateLimitError),
        reraise=True,
    )
    async def _call_claude_api(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> ClaudeResponse:
        """
        Make API call to Claude via OpenRouter with retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            ClaudeResponse with parsed response data
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/fluffypony/mcp-code-indexer",
            "X-Title": "MCP Code Indexer",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        self.logger.info("Sending request to Claude API via OpenRouter...")
        self.logger.info(f"  Model: {self.config.model}")
        self.logger.info(f"  Temperature: {self.config.temperature}")
        self.logger.info(f"  Max tokens: {self.config.max_tokens}")
        self.logger.info(f"  Timeout: {self.config.timeout}s")

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.OPENROUTER_API_URL, headers=headers, json=payload
                ) as response:

                    self.logger.info(f"Claude API response status: {response.status}")

                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(
                            f"Rate limited by OpenRouter, retry after {retry_after}s"
                        )
                        raise ClaudeRateLimitError(
                            f"Rate limited. Retry after {retry_after}s", retry_after
                        )

                    response.raise_for_status()

                    response_data = await response.json()

                    if "choices" not in response_data:
                        self.logger.error(
                            f"Invalid API response format: {response_data}"
                        )
                        raise ClaudeAPIError(
                            f"Invalid API response format: {response_data}"
                        )

                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage")
                    model = response_data.get("model")

                    self.logger.info(
                        f"Claude response content length: {len(content)} characters"
                    )
                    if usage:
                        self.logger.info(f"Token usage: {usage}")

                    return ClaudeResponse(content=content, usage=usage, model=model)

        except aiohttp.ClientError as e:
            self.logger.error(f"Claude API request failed: {e}")
            raise ClaudeAPIError(f"Claude API request failed: {e}")
        except asyncio.TimeoutError:
            self.logger.error(
                f"Claude API request timed out after {self.config.timeout}s"
            )
            raise ClaudeAPIError("Claude API request timed out")

    def validate_json_response(
        self, response_text: str, required_keys: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and parse JSON response from Claude.

        Args:
            response_text: Raw response content
            required_keys: List of required keys in the JSON response

        Returns:
            Validated JSON data
        """

        def extract_json_from_response(text: str) -> str:
            """Extract JSON from response that might have extra text before/after."""
            text = text.strip()

            # Try to find JSON in the response
            json_start = -1
            json_end = -1

            # Look for opening brace
            for i, char in enumerate(text):
                if char == "{":
                    json_start = i
                    break

            if json_start == -1:
                return text  # No JSON found, return original

            # Find matching closing brace
            brace_count = 0
            for i in range(json_start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end == -1:
                return text  # No matching brace found, return original

            return text[json_start:json_end]

        try:
            # First try parsing as-is
            try:
                data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                # Try extracting JSON from response
                extracted_json = extract_json_from_response(response_text)
                if extracted_json != response_text.strip():
                    self.logger.debug(f"Extracted JSON from response: {extracted_json}")
                data = json.loads(extracted_json)

            # Validate required keys if specified
            if required_keys:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    raise ClaudeValidationError(
                        f"Missing required keys in response: {missing_keys}"
                    )

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response text: {response_text}")
            raise ClaudeValidationError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            raise ClaudeValidationError(f"Response validation failed: {e}")

    def format_error_response(self, error: Exception, context: str = "") -> str:
        """
        Format error for user-friendly display.

        Args:
            error: The exception that occurred
            context: Additional context about the operation

        Returns:
            Formatted error message
        """
        if isinstance(error, ClaudeRateLimitError):
            return (
                f"Rate limited by Claude API. Please wait {error.retry_after} "
                "seconds and try again."
            )
        elif isinstance(error, ClaudeValidationError):
            return f"Invalid response from Claude API: {str(error)}"
        elif isinstance(error, ClaudeAPIError):
            return f"Claude API error: {str(error)}"
        else:
            return f"Unexpected error during {context}: {str(error)}"

    async def find_existing_project_by_name(self, project_name: str) -> Optional[Any]:
        """
        Find existing project by name for CLI usage.

        Args:
            project_name: Name of the project to find

        Returns:
            Project object if found, None otherwise
        """
        try:
            all_projects = await self.db_manager.get_all_projects()
            normalized_name = project_name.lower()

            for project in all_projects:
                if project.name.lower() == normalized_name:
                    self.logger.info(
                        f"Found existing project: {project.name} (ID: {project.id})"
                    )
                    return project

            self.logger.warning(f"No existing project found with name: {project_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error finding project by name: {e}")
            return None

    async def get_project_overview(self, project_info: Dict[str, str]) -> str:
        """
        Get project overview from database.

        Args:
            project_info: Project information dict with projectName, folderPath, etc.

        Returns:
            Project overview text or empty string if not found
        """
        try:
            # Try to find existing project by name first
            project = await self.find_existing_project_by_name(
                project_info["projectName"]
            )

            if not project:
                self.logger.warning(
                    f"Project '{project_info['projectName']}' not found in database"
                )
                return ""

            # Get overview for the project using project.id
            overview_result = await self.db_manager.get_project_overview(project.id)
            if overview_result:
                return overview_result.overview
            else:
                return ""
        except Exception as e:
            self.logger.warning(f"Failed to get project overview: {e}")
            return ""
