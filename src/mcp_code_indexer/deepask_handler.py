#!/usr/bin/env python3
"""
DeepAsk Handler for MCP Code Indexer

Handles enhanced question-answering with two-stage processing:
1. Extract search terms and compress overview
2. Search file descriptions and provide enhanced answer
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .claude_api_handler import ClaudeAPIHandler, ClaudeAPIError
from .database.database import DatabaseManager


class DeepAskError(ClaudeAPIError):
    """Exception specific to DeepAsk operations."""

    pass


class DeepAskHandler(ClaudeAPIHandler):
    """
    Handler for enhanced Q&A operations using two-stage Claude API processing.

    Stage 1: Extract search terms and compress project overview
    Stage 2: Search file descriptions and provide enhanced answer with context
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize DeepAskHandler.

        Args:
            db_manager: Database manager instance
            cache_dir: Cache directory for temporary files
            logger: Logger instance to use (optional, creates default if not provided)
        """
        super().__init__(db_manager, cache_dir, logger)
        self.logger = logger if logger is not None else logging.getLogger(__name__)

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

    async def deepask_question(
        self, project_info: Dict[str, str], question: str, max_file_results: int = 10
    ) -> Dict[str, Any]:
        """
        Ask an enhanced question about the project using two-stage Claude API
        processing.

        Args:
            project_info: Project information dict with projectName, folderPath, etc.
            question: User's question about the project
            max_file_results: Maximum number of file descriptions to include

        Returns:
            Dict containing enhanced response and metadata
        """
        try:
            self.logger.info(
                f"Processing deepask question for project: "
                f"{project_info['projectName']}"
            )
            self.logger.info(f"Question: {question}")

            # Validate inputs
            if not question or not question.strip():
                raise DeepAskError("Question cannot be empty")

            if not project_info.get("projectName"):
                raise DeepAskError("Project name is required")

            # Stage 1: Extract search terms and compress overview
            stage1_result = await self._stage1_extract_search_terms(
                project_info, question
            )

            # Stage 2: Search files and provide enhanced answer
            stage2_result = await self._stage2_enhanced_answer(
                project_info,
                question,
                stage1_result["search_terms"],
                stage1_result["compressed_overview"],
                max_file_results,
            )

            # Combine results
            result = {
                "answer": stage2_result["answer"],
                "project_name": project_info["projectName"],
                "question": question,
                "search_terms": stage1_result["search_terms"],
                "compressed_overview": stage1_result["compressed_overview"],
                "relevant_files": stage2_result["relevant_files"],
                "metadata": {
                    "model": self.config.model,
                    "stage1_tokens": stage1_result["token_usage"],
                    "stage2_tokens": stage2_result["token_usage"],
                    "total_files_found": stage2_result["total_files_found"],
                    "files_included": len(stage2_result["relevant_files"]),
                },
            }

            self.logger.info("DeepAsk question completed successfully")
            self.logger.info(f"Search terms: {stage1_result['search_terms']}")
            self.logger.info(f"Files found: {stage2_result['total_files_found']}")
            self.logger.info(f"Files included: {len(stage2_result['relevant_files'])}")

            return result

        except Exception as e:
            error_msg = f"Failed to process deepask question: {str(e)}"
            self.logger.error(error_msg)
            if isinstance(e, (ClaudeAPIError, DeepAskError)):
                raise
            else:
                raise DeepAskError(error_msg)

    async def _stage1_extract_search_terms(
        self, project_info: Dict[str, str], question: str
    ) -> Dict[str, Any]:
        """
        Stage 1: Extract search terms and compress project overview.

        Args:
            project_info: Project information
            question: User's question

        Returns:
            Dict with search_terms, compressed_overview, and token_usage
        """
        self.logger.info("Stage 1: Extracting search terms and compressing overview")

        # Get project overview
        overview = await self.get_project_overview(project_info)
        if not overview:
            overview = "No project overview available."

        # Build stage 1 prompt
        prompt = self._build_stage1_prompt(project_info, question, overview)

        # Validate token limits for stage 1
        if not self.validate_token_limit(prompt):
            raise DeepAskError(
                f"Stage 1 prompt exceeds token limit of {self.config.token_limit}. "
                "Project overview may be too large."
            )

        # Call Claude API for stage 1
        system_prompt = self._get_stage1_system_prompt()
        response = await self._call_claude_api(prompt, system_prompt)

        # Parse and validate response
        response_data = self.validate_json_response(
            response.content, required_keys=["search_terms", "compressed_overview"]
        )

        token_usage = {
            "prompt_tokens": self.get_token_count(prompt),
            "response_tokens": (
                response.usage.get("completion_tokens") if response.usage else None
            ),
            "total_tokens": (
                response.usage.get("total_tokens") if response.usage else None
            ),
        }

        return {
            "search_terms": response_data["search_terms"],
            "compressed_overview": response_data["compressed_overview"],
            "token_usage": token_usage,
        }

    async def _stage2_enhanced_answer(
        self,
        project_info: Dict[str, str],
        question: str,
        search_terms: List[str],
        compressed_overview: str,
        max_file_results: int,
    ) -> Dict[str, Any]:
        """
        Stage 2: Search file descriptions and provide enhanced answer.

        Args:
            project_info: Project information
            question: User's question
            search_terms: Search terms from stage 1
            compressed_overview: Compressed overview from stage 1
            max_file_results: Maximum number of files to include

        Returns:
            Dict with answer, relevant_files, total_files_found, and token_usage
        """
        self.logger.info("Stage 2: Searching files and generating enhanced answer")
        self.logger.info(f"Search terms: {search_terms}")

        # Search for relevant files
        relevant_files = []
        total_files_found = 0

        try:
            # Find existing project by name only (don't create new ones for Q&A)
            project = await self.find_existing_project_by_name(
                project_info["projectName"]
            )

            if not project:
                self.logger.warning(
                    f"Project '{project_info['projectName']}' not found in database"
                )
                return {
                    "answer": (
                        f"Project '{project_info['projectName']}' not found in "
                        f"database. Please check the project name."
                    ),
                    "relevant_files": [],
                    "total_files_found": 0,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "total_tokens": 0,
                    },
                }

            for search_term in search_terms:
                try:
                    search_results = await self.db_manager.search_file_descriptions(
                        project_id=project.id,
                        query=search_term,
                        max_results=max_file_results,
                    )

                    total_files_found += len(search_results)

                    # Add unique files to relevant_files
                    for result in search_results:
                        if not any(
                            f["filePath"] == result.file_path for f in relevant_files
                        ):
                            relevant_files.append(
                                {
                                    "filePath": result.file_path,
                                    "description": result.description,
                                    "search_term": search_term,
                                    "relevance_score": result.relevance_score,
                                }
                            )

                            # Stop if we have enough files
                            if len(relevant_files) >= max_file_results:
                                break

                    if len(relevant_files) >= max_file_results:
                        break

                except Exception as e:
                    self.logger.warning(f"Search failed for term '{search_term}': {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Failed to search files: {e}")
            # Continue with empty relevant_files list

        # Build stage 2 prompt with file context
        prompt = self._build_stage2_prompt(
            project_info, question, compressed_overview, relevant_files
        )

        # Validate token limits for stage 2
        if not self.validate_token_limit(prompt):
            # Try reducing file context
            self.logger.warning(
                "Stage 2 prompt exceeds token limit, reducing file context"
            )
            reduced_files = relevant_files[: max_file_results // 2]
            prompt = self._build_stage2_prompt(
                project_info, question, compressed_overview, reduced_files
            )

            if not self.validate_token_limit(prompt):
                raise DeepAskError(
                    "Stage 2 prompt still exceeds token limit even with reduced "
                    "context. Try a more specific question."
                )

            relevant_files = reduced_files

        # Call Claude API for stage 2
        system_prompt = self._get_stage2_system_prompt()
        response = await self._call_claude_api(prompt, system_prompt)

        token_usage = {
            "prompt_tokens": self.get_token_count(prompt),
            "response_tokens": (
                response.usage.get("completion_tokens") if response.usage else None
            ),
            "total_tokens": (
                response.usage.get("total_tokens") if response.usage else None
            ),
        }

        return {
            "answer": response.content,
            "relevant_files": relevant_files,
            "total_files_found": total_files_found,
            "token_usage": token_usage,
        }

    def _build_stage1_prompt(
        self, project_info: Dict[str, str], question: str, overview: str
    ) -> str:
        """Build stage 1 prompt for extracting search terms."""
        project_name = project_info["projectName"]

        return f"""I need to answer a question about the codebase "{project_name}".
To provide the best answer, I need to search for relevant files and then answer
the question.

PROJECT OVERVIEW:
{overview}

QUESTION:
{question}

Please analyze the question and project overview, then provide:

1. A list of 3-5 search terms that would help find relevant files to answer
   this question
2. A compressed version of the project overview (2-3 sentences max) that
   captures the most relevant information for this question

Respond with valid JSON in this format:
{{
  "search_terms": ["term1", "term2", "term3"],
  "compressed_overview": "Brief summary focusing on aspects relevant to the question..."
}}"""

    def _build_stage2_prompt(
        self,
        project_info: Dict[str, str],
        question: str,
        compressed_overview: str,
        relevant_files: List[Dict[str, Any]],
    ) -> str:
        """Build stage 2 prompt for enhanced answer."""
        project_name = project_info["projectName"]

        # Format file descriptions
        file_context = ""
        if relevant_files:
            file_context = "\n\nRELEVANT FILES:\n"
            for i, file_info in enumerate(relevant_files, 1):
                file_context += f"\n{i}. {file_info['filePath']}\n"
                file_context += f"   Description: {file_info['description']}\n"
                file_context += f"   Found via search: {file_info['search_term']}\n"
        else:
            file_context = "\n\nNo relevant files found in the search."

        return (
            f"Please answer the following question about the codebase "
            f'"{project_name}".\n\n'
            f"PROJECT OVERVIEW (COMPRESSED):\n{compressed_overview}\n{file_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Please provide a comprehensive answer based on the project overview and "
            "relevant file descriptions above. Reference specific files when "
            "appropriate and explain how they relate to the question. If the "
            "available information is insufficient, clearly state what "
            "additional details would be needed."
        )

    def _get_stage1_system_prompt(self) -> str:
        """Get system prompt for stage 1."""
        return """You are a technical assistant that analyzes software projects to
extract relevant search terms and compress information.

Your task:
1. Analyze the user's question about a codebase
2. Extract 3-5 search terms that would help find relevant files to answer the question
3. Compress the project overview to focus on information relevant to the question

Search terms should be:
- Technical keywords (function names, class names, concepts)
- File types or directory names if relevant
- Domain-specific terminology from the question

The compressed overview should:
- Be 2-3 sentences maximum
- Focus only on aspects relevant to answering the question
- Preserve the most important architectural or functional details

Always respond with valid JSON matching the requested format."""

    def _get_stage2_system_prompt(self) -> str:
        """Get system prompt for stage 2."""
        return """You are a software engineering expert that provides detailed
answers about codebases using available context.

When answering:
1. Use the compressed project overview for high-level context
2. Reference specific files from the relevant files list when they
   relate to the question
3. Explain how different files work together if relevant
4. Be specific and technical when appropriate
5. If information is incomplete, clearly state what's missing and suggest next steps
6. Provide actionable insights when possible

Your answer should be comprehensive but focused on the specific question asked."""

    def format_response(self, result: Dict[str, Any], format_type: str = "text") -> str:
        """
        Format response for CLI output.

        Args:
            result: Result from deepask_question
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

        # Show search terms used
        output.append(f"Search terms: {', '.join(result['search_terms'])}")
        output.append("")

        # Show relevant files
        if result["relevant_files"]:
            output.append("Relevant files analyzed:")
            for i, file_info in enumerate(result["relevant_files"], 1):
                output.append(f"  {i}. {file_info['filePath']}")
        else:
            output.append("No relevant files found.")
        output.append("")

        # Show metadata
        output.append("Metadata:")
        output.append(f"  Model: {metadata['model']}")
        output.append(f"  Total files found: {metadata['total_files_found']}")
        output.append(f"  Files included: {metadata['files_included']}")

        stage1_tokens = metadata["stage1_tokens"]["total_tokens"]
        stage2_tokens = metadata["stage2_tokens"]["total_tokens"]
        if stage1_tokens and stage2_tokens:
            output.append(
                (
                    f"  Total tokens: {stage1_tokens + stage2_tokens} "
                    f"(Stage 1: {stage1_tokens}, Stage 2: {stage2_tokens})"
                )
            )

        return "\n".join(output)
