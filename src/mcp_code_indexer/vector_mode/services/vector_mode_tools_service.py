"""
Vector Mode Tools Service for similarity search operations.

Orchestrates ASTChunker, EmbeddingService, and VectorStorageService to provide
find_similar_code functionality for both code snippets and file sections.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from turbopuffer.types import Row

from .. import is_vector_mode_available
from ..chunking.ast_chunker import ASTChunker, CodeChunk
from ..config import VectorConfig, load_vector_config
from ..const import MODEL_DIMENSIONS
from ..providers.voyage_client import VoyageClient
from ..providers.turbopuffer_client import TurbopufferClient
from .embedding_service import EmbeddingService
from .vector_storage_service import VectorStorageService

logger = logging.getLogger(__name__)


class VectorModeToolsService:
    """
    Service for vector-based code similarity search operations.

    Provides find_similar_code functionality that handles both code snippet
    and file section inputs, orchestrating chunking, embedding, and search.
    """

    def __init__(self):
        """Initialize VectorModeToolsService and set up vector mode dependencies."""
        self.ast_chunker: Optional[ASTChunker] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_storage_service: Optional[VectorStorageService] = None
        self.config: Optional[VectorConfig] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure vector mode services are initialized."""
        if self._initialized:
            return

        # Check if vector mode dependencies are available
        if not is_vector_mode_available():
            raise RuntimeError("Vector mode dependencies are not available")

        # Load vector configuration
        self.config = load_vector_config()

        # Initialize clients
        voyage_client = VoyageClient(
            api_key=self.config.voyage_api_key, model=self.config.embedding_model
        )

        turbopuffer_client = TurbopufferClient(
            api_key=self.config.turbopuffer_api_key,
            region=self.config.turbopuffer_region,
        )

        # Get embedding dimension for the model
        embedding_dimension = MODEL_DIMENSIONS.get(self.config.embedding_model, 1024)

        # Initialize services
        self.ast_chunker = ASTChunker()
        self.embedding_service = EmbeddingService(voyage_client, self.config)
        self.vector_storage_service = VectorStorageService(
            turbopuffer_client, embedding_dimension, self.config
        )

        self._initialized = True

        logger.info(
            "Vector mode services initialized",
            extra={
                "structured_data": {
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "batch_size": self.config.batch_size,
                }
            },
        )

    async def find_similar_code(
        self,
        project_name: str,
        folder_path: str,
        code_snippet: Optional[str] = None,
        file_path: Optional[str] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find code similar to a given snippet or file section.

        Args:
            project_name: Name of the project to search in
            folder_path: Root folder path of the project
            code_snippet: Direct code snippet to search for (mutually exclusive with file_path)
            file_path: Path to file containing code to search for (requires line_start/line_end)
            line_start: Starting line number for file section (1-indexed)
            line_end: Ending line number for file section (1-indexed)
            similarity_threshold: Minimum similarity score (defaults to config value)
            max_results: Maximum number of results (defaults to config value)

        Returns:
            Dictionary containing search results and metadata

        Raises:
            ValueError: If input validation fails
            RuntimeError: If search operations fail
        """
        # Validate mutually exclusive inputs
        if code_snippet and file_path:
            raise ValueError("Cannot specify both code_snippet and file_path")

        if not code_snippet and not file_path:
            raise ValueError(
                "Must specify either code_snippet or file_path with line range"
            )

        if file_path and (line_start is None or line_end is None):
            raise ValueError("file_path requires both line_start and line_end")

        # Ensure services are initialized
        self._ensure_initialized()

        # Use config defaults if not specified
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        max_results = max_results or self.config.max_search_results

        logger.info(
            "Starting code similarity search",
            extra={
                "structured_data": {
                    "project_name": project_name,
                    "has_code_snippet": bool(code_snippet),
                    "file_path": file_path,
                    "line_range": f"{line_start}-{line_end}" if line_start else None,
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results,
                }
            },
        )

        try:
            # Get query code content
            if code_snippet:
                query_code = code_snippet
                query_source = "code_snippet"
            else:
                query_code = await self._read_file_section(
                    folder_path, file_path, line_start, line_end
                )
                query_source = f"{file_path}:{line_start}-{line_end}"

            # Chunk the query code
            query_chunks = await self._chunk_code(query_code, query_source)

            if not query_chunks:
                logger.warning(
                    "No chunks generated from query code",
                    extra={"structured_data": {"query_source": query_source}},
                )
                return {
                    "results": [],
                    "total_results": 0,
                    "query_info": {
                        "source": query_source,
                        "chunks_generated": 0,
                        "similarity_threshold": similarity_threshold,
                    },
                    "message": "No valid code chunks could be generated from the input",
                }

            # Generate embeddings for query chunks
            query_embeddings = (
                await self.embedding_service.generate_embeddings_for_chunks(
                    query_chunks, project_name, Path(query_source)
                )
            )

            # Search for similar code using each query chunk
            # Get more results per chunk to allow for deduplication and filtering
            results_per_chunk = min(
                max_results * 2, 50
            )  # Cap at 50 per chunk to avoid excessive results

            all_results: List[Row] = []
            for i, (chunk, embedding) in enumerate(zip(query_chunks, query_embeddings)):
                logger.debug(
                    f"Searching with query chunk {i+1}/{len(query_chunks)}",
                    extra={
                        "structured_data": {
                            "chunk_type": chunk.chunk_type.value,
                            "chunk_name": chunk.name,
                        }
                    },
                )

                chunk_results = await self.vector_storage_service.search_similar_chunks(
                    query_embedding=embedding,
                    project_name=project_name,
                    top_k=results_per_chunk,
                )

                if chunk_results:
                    all_results.extend(chunk_results)

            # Filter by similarity threshold and deduplicate
            filtered_results = self._filter_and_deduplicate_results(
                all_results, similarity_threshold, max_results
            )

            logger.info(
                "Code similarity search completed",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "query_chunks": len(query_chunks),
                        "raw_results": len(all_results),
                        "filtered_results": len(filtered_results),
                        "similarity_threshold": similarity_threshold,
                    }
                },
            )

            return {
                "results": filtered_results,
                "total_results": len(filtered_results),
                "query_info": {
                    "source": query_source,
                    "chunks_generated": len(query_chunks),
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results,
                },
            }

        except Exception as e:
            logger.error(
                "Code similarity search failed",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "query_source": (
                            query_source if "query_source" in locals() else "unknown"
                        ),
                        "error": str(e),
                    }
                },
                exc_info=True,
            )
            raise RuntimeError(f"Similarity search failed: {e}") from e

    async def _read_file_section(
        self, folder_path: str, file_path: str, line_start: int, line_end: int
    ) -> str:
        """
        Read a specific section of a file.

        Args:
            folder_path: Root folder path
            file_path: Relative path to the file
            line_start: Starting line number (1-indexed)
            line_end: Ending line number (1-indexed, inclusive)

        Returns:
            Content of the specified file section

        Raises:
            ValueError: If file path is invalid or lines are out of range
            RuntimeError: If file cannot be read
        """
        try:
            # Resolve file path safely
            folder_path_obj = Path(folder_path).expanduser().resolve()
            file_path_obj = folder_path_obj / file_path

            # Security check: ensure file is within project folder
            if not str(file_path_obj.resolve()).startswith(str(folder_path_obj)):
                raise ValueError(f"File path {file_path} is outside project folder")

            if not file_path_obj.exists():
                raise ValueError(f"File not found: {file_path}")

            # Read file content
            content = file_path_obj.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            # Validate line range
            if line_start < 1 or line_end < 1:
                raise ValueError("Line numbers must be >= 1")

            if line_start > len(lines):
                raise ValueError(
                    f"line_start {line_start} exceeds file length {len(lines)}"
                )

            # Clamp line_end to file length
            actual_line_end = min(line_end, len(lines))
            if line_end > len(lines):
                logger.warning(
                    f"line_end {line_end} exceeds file length {len(lines)}, clamping to {actual_line_end}"
                )

            # Extract section (convert to 0-based indexing)
            section_lines = lines[line_start - 1 : actual_line_end]
            return "\n".join(section_lines)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to read file section {file_path}:{line_start}-{line_end}: {e}"
            ) from e

    async def _chunk_code(
        self, code_content: str, source_identifier: str
    ) -> List[CodeChunk]:
        """
        Chunk code content using AST-based analysis.

        Args:
            code_content: Code to chunk
            source_identifier: Identifier for logging (e.g., filename or "code_snippet")

        Returns:
            List of code chunks

        Raises:
            RuntimeError: If chunking fails
        """
        try:
            # Attempt to detect language from source or content
            language = "python"  # Default fallback
            # TODO: remove. This is handle by language handler in ASTChunker
            if "." in source_identifier:
                ext = Path(source_identifier).suffix.lower()
                if ext in [".py"]:
                    language = "python"
                elif ext in [".js", ".ts"]:
                    language = "javascript"
                elif ext in [".java"]:
                    language = "java"
                elif ext in [".cpp", ".cc", ".cxx"]:
                    language = "cpp"
                elif ext in [".c"]:
                    language = "c"

            # Run in executor to avoid blocking the event loop (CPU-bound work)
            loop = asyncio.get_running_loop()

            def do_chunk() -> List[CodeChunk]:
                return self.ast_chunker.chunk_content(
                    content=code_content,
                    file_path=source_identifier,
                    language=language,
                )

            chunks = await loop.run_in_executor(None, do_chunk)

            logger.debug(
                f"Generated {len(chunks)} chunks from {source_identifier}",
                extra={
                    "structured_data": {
                        "source": source_identifier,
                        "language": language,
                        "chunk_count": len(chunks),
                    }
                },
            )

            return chunks

        except Exception as e:
            logger.error(
                f"Failed to chunk code from {source_identifier}: {e}",
                extra={"structured_data": {"source": source_identifier}},
                exc_info=True,
            )
            raise RuntimeError(f"Code chunking failed: {e}") from e

    def _filter_and_deduplicate_results(
        self,
        results: List[Row],
        similarity_threshold: float,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter results by similarity threshold and remove duplicates.

        Args:
            results: Raw search results (turbopuffer Row objects)
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results to return

        Returns:
            Filtered and deduplicated results as dictionaries
        """
        processed_results = []

        for row in results:
            if row is None:
                continue

            # Extract similarity score (turbopuffer uses $dist for distance)
            # Lower distance = higher similarity, so convert: similarity = 1 - distance
            distance = getattr(row, "$dist", 1.0)
            similarity = 1.0 - distance

            # Filter by similarity threshold
            if similarity < similarity_threshold:
                continue

            # Convert to result dictionary
            file_path = getattr(row, "file_path", "")
            file_name = Path(file_path).name if file_path else ""

            result_dict = {
                "file_name": file_name,
                "start_line": getattr(row, "start_line", 0),
                "end_line": getattr(row, "end_line", 0),
                "score": similarity,
                "content": getattr(row, "content", ""),
                "metadata": {
                    "file_path": file_path,
                    "content_hash": getattr(row, "content_hash", ""),
                    "chunk_type": getattr(row, "chunk_type", ""),
                },
            }
            processed_results.append(result_dict)

        # Sort by similarity score (descending)
        processed_results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate by file_path + content hash to avoid duplicate chunks
        seen = set()
        deduplicated = []

        for result in processed_results:
            metadata = result["metadata"]
            file_path = metadata["file_path"]
            content_hash = metadata["content_hash"]
            dedup_key = f"{file_path}:{content_hash}"

            if dedup_key not in seen:
                seen.add(dedup_key)
                deduplicated.append(result)

                if len(deduplicated) >= max_results:
                    break

        return deduplicated
