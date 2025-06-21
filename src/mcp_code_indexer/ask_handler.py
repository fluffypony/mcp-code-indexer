#!/usr/bin/env python3
"""
Ask Handler for MCP Code Indexer

Handles simple question-answering by combining project overview with user questions
and sending them to Claude via OpenRouter API for direct responses.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .claude_api_handler import ClaudeAPIHandler, ClaudeAPIError
from .database.database import DatabaseManager


class AskError(ClaudeAPIError):
    """Exception specific to Ask operations."""

    pass


class AskHandler(ClaudeAPIHandler):
    """
    Handler for simple Q&A operations using Claude API.

    Provides functionality to:
    - Combine project overview with user questions
    - Send combined prompt to Claude for analysis
    - Return formatted responses for CLI consumption
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize AskHandler.

        Args:
            db_manager: Database manager instance
            cache_dir: Cache directory for temporary files
            logger: Logger instance to use (optional, creates default if not provided)
        """
        super().__init__(db_manager, cache_dir, logger)
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    async def ask_question(
        self, project_info: Dict[str, str], question: str, include_overview: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question about the project using Claude API.

        Args:
            project_info: Project information dict with projectName, folderPath, etc.
            question: User's question about the project
            include_overview: Whether to include project overview in context

        Returns:
            Dict containing response and metadata
        """
        try:
            self.logger.info(
                f"Processing ask question for project: {project_info['projectName']}"
            )
            self.logger.info(f"Question: {question}")

            # Validate inputs
            if not question or not question.strip():
                raise AskError("Question cannot be empty")

            if not project_info.get("projectName"):
                raise AskError("Project name is required")

            # Get project overview if requested
            overview = ""
            if include_overview:
                overview = await self.get_project_overview(project_info)
                if not overview:
                    self.logger.warning(
                        f"No project overview found for {project_info['projectName']}"
                    )
                    overview = "No project overview available."

            # Build the prompt
            prompt = self._build_ask_prompt(project_info, question, overview)

            # Validate token limits
            if not self.validate_token_limit(prompt):
                raise AskError(
                    f"Question and project context exceed token limit of "
                    f"{self.config.token_limit}. Please ask a more specific "
                    "question or use --deepask for enhanced search."
                )

            # Get token counts for reporting
            overview_tokens = self.get_token_count(overview) if overview else 0
            question_tokens = self.get_token_count(question)
            total_prompt_tokens = self.get_token_count(prompt)

            self.logger.info(
                f"Token usage: overview={overview_tokens}, "
                f"question={question_tokens}, total={total_prompt_tokens}"
            )

            # Call Claude API
            system_prompt = self._get_system_prompt()
            response = await self._call_claude_api(prompt, system_prompt)

            # Format response
            result = {
                "answer": response.content,
                "project_name": project_info["projectName"],
                "question": question,
                "metadata": {
                    "model": response.model or self.config.model,
                    "token_usage": {
                        "overview_tokens": overview_tokens,
                        "question_tokens": question_tokens,
                        "total_prompt_tokens": total_prompt_tokens,
                        "response_tokens": (
                            response.usage.get("completion_tokens")
                            if response.usage
                            else None
                        ),
                        "total_tokens": (
                            response.usage.get("total_tokens")
                            if response.usage
                            else None
                        ),
                    },
                    "include_overview": include_overview,
                },
            }

            self.logger.info("Ask question completed successfully")
            return result

        except Exception as e:
            error_msg = f"Failed to process ask question: {str(e)}"
            self.logger.error(error_msg)
            if isinstance(e, (ClaudeAPIError, AskError)):
                raise
            else:
                raise AskError(error_msg)

    def _build_ask_prompt(
        self, project_info: Dict[str, str], question: str, overview: str
    ) -> str:
        """
        Build the prompt for Claude API.

        Args:
            project_info: Project information
            question: User's question
            overview: Project overview (may be empty)

        Returns:
            Formatted prompt string
        """
        project_name = project_info["projectName"]

        if overview.strip():
            prompt = (
                f"Please answer the following question about the codebase "
                f'"{project_name}".\n\n'
                f"PROJECT OVERVIEW:\n{overview}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"Please provide a clear, detailed answer based on the project "
                f"overview above. If the overview doesn't contain enough "
                f"information to fully answer the question, please say so and "
                f"suggest what additional information might be needed."
            )
        else:
            prompt = (
                f"Please answer the following question about the codebase "
                f'"{project_name}".\n\n'
                f"Note: No project overview is available for this codebase.\n\n"
                f"QUESTION:\n{question}\n\n"
                f"Please provide the best answer you can based on the project "
                f"name and general software development knowledge. If you need "
                f"more specific information about this codebase to provide a "
                f"complete answer, please mention what would be helpful."
            )

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for Claude API."""
        return (
            "You are a helpful software engineering assistant that analyzes "
            "codebases and answers questions about them.\n\n"
            "When answering questions:\n"
            "1. Be specific and technical when appropriate\n"
            "2. Reference the project overview when available\n"
            "3. If information is missing, clearly state what you don't know\n"
            "4. Provide actionable suggestions when possible\n"
            "5. Use clear, professional language\n"
            "6. Focus on the specific question asked\n\n"
            "If the project overview is insufficient to answer the question "
            "completely, explain what additional information would be needed "
            "and suggest using --deepask for more detailed analysis."
        )

    def format_response(self, result: Dict[str, Any], format_type: str = "text") -> str:
        """
        Format response for CLI output.

        Args:
            result: Result from ask_question
            format_type: Output format ("text" or "json")

        Returns:
            Formatted response string
        """
        if format_type == "json":
            import json

            return json.dumps(result, indent=2)

        # Text format
        answer = result["answer"]
        metadata = result["metadata"]

        output = []
        output.append(f"Question: {result['question']}")
        output.append(f"Project: {result['project_name']}")
        output.append("")
        output.append("Answer:")
        output.append(answer)
        output.append("")
        output.append("Metadata:")
        output.append(f"  Model: {metadata['model']}")
        output.append(f"  Overview included: {metadata['include_overview']}")

        if metadata["token_usage"]["total_tokens"]:
            output.append(f"  Total tokens: {metadata['token_usage']['total_tokens']}")
        else:
            output.append(
                f"  Prompt tokens: {metadata['token_usage']['total_prompt_tokens']}"
            )

        return "\n".join(output)
