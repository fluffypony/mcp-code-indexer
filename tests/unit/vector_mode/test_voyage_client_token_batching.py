"""
Unit tests for VoyageClient token-based batching functionality.

Tests the token-based batching logic in generate_embeddings_batch method
to ensure it respects the Voyage API token limits.
"""

from typing import List, Tuple
from unittest.mock import Mock, patch
import pytest

from mcp_code_indexer.vector_mode.providers.voyage_client import VoyageClient


class TestVoyageClientTokenBatching:
    """Test VoyageClient token-based batching functionality."""

    @pytest.fixture
    def voyage_client(self):
        """Create a VoyageClient with mocked actual client."""
        with patch('voyageai.Client'):
            client = VoyageClient(api_key="test-key", model="voyage-code-2")
            # Mock the underlying client to avoid actual API calls
            client.client = Mock()
            return client

    def test_generate_embeddings_batch_single_batch(self, voyage_client):
        """Test batch processing when tokens are under the limit."""
        # Prepare test data
        all_texts = ["short text", "another short text"]
        file_boundaries = [("file1.py", 0, 1), ("file2.py", 1, 2)]
        
        # Mock estimate_cost to return tokens under limit
        voyage_client.estimate_cost = Mock(return_value={
            "total_tokens": 1000,  # Under default 120k limit
            "total_texts": 2
        })
        
        # Mock single embedding generation
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        voyage_client.generate_embeddings = Mock(return_value=mock_embeddings)
        
        result = voyage_client.generate_embeddings_batch(
            all_texts=all_texts,
            file_boundaries=file_boundaries,
            max_tokens_per_batch=120000
        )
        
        # Verify single API call was made
        voyage_client.generate_embeddings.assert_called_once_with(
            all_texts, input_type="document"
        )
        
        # Verify correct file grouping
        expected_result = {
            "file1.py": [[0.1, 0.2]],
            "file2.py": [[0.3, 0.4]]
        }
        assert result == expected_result

    def test_generate_embeddings_batch_token_splitting(self, voyage_client):
        """Test batch processing when tokens exceed the limit."""
        # Create larger texts that would exceed token limit
        all_texts = ["x" * 250000, "y" * 250000]  # Each ~62.5k tokens (250k chars / 4)
        file_boundaries = [("file1.py", 0, 1), ("file2.py", 1, 2)]
        
        # Mock estimate_cost to return tokens over limit
        voyage_client.estimate_cost = Mock(return_value={
            "total_tokens": 125000,  # Over 120k limit
            "total_texts": 2
        })
        
        # Mock the token batching method
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        voyage_client._generate_embeddings_with_token_batching = Mock(
            return_value=expected_embeddings
        )
        
        result = voyage_client.generate_embeddings_batch(
            all_texts=all_texts,
            file_boundaries=file_boundaries,
            max_tokens_per_batch=120000
        )
        
        # Verify token batching method was called
        voyage_client._generate_embeddings_with_token_batching.assert_called_once_with(
            all_texts, 120000, "document"
        )
        
        # Verify correct file grouping
        expected_result = {
            "file1.py": [[0.1, 0.2]],
            "file2.py": [[0.3, 0.4]]
        }
        assert result == expected_result

    def test_generate_embeddings_with_token_batching(self, voyage_client):
        """Test the token-based batching logic directly."""
        # Create texts with known token counts
        # Using 4 chars per token estimate: 
        # text1: 240 chars = 60 tokens
        # text2: 240 chars = 60 tokens  
        # text3: 240 chars = 60 tokens
        text1 = "x" * 240  # ~60 tokens
        text2 = "y" * 240  # ~60 tokens  
        text3 = "z" * 240  # ~60 tokens
        all_texts = [text1, text2, text3]
        
        # Set token limit to 100 - should batch as [text1, text2] (120 tokens > 100, so text1 alone), then [text2], then [text3]
        # Actually, let's make it clearer: limit=130 should allow [text1, text2] (120 tokens), then [text3] (60 tokens)
        max_tokens_per_batch = 130
        
        # Mock generate_embeddings to return different embeddings per batch
        voyage_client.generate_embeddings = Mock(side_effect=[
            [[0.1, 0.2], [0.3, 0.4]],  # First batch: text1, text2
            [[0.5, 0.6]]               # Second batch: text3
        ])
        
        result = voyage_client._generate_embeddings_with_token_batching(
            all_texts, max_tokens_per_batch, "document"
        )
        
        # Verify two API calls were made
        assert voyage_client.generate_embeddings.call_count == 2
        
        # Verify first batch had texts 1 and 2
        first_call = voyage_client.generate_embeddings.call_args_list[0]
        assert first_call[0][0] == [text1, text2]
        assert first_call[1]["input_type"] == "document"
        
        # Verify second batch had text 3
        second_call = voyage_client.generate_embeddings.call_args_list[1]
        assert second_call[0][0] == [text3]
        assert second_call[1]["input_type"] == "document"
        
        # Verify all embeddings were combined
        expected_result = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert result == expected_result

    def test_generate_embeddings_with_token_batching_single_large_text(self, voyage_client):
        """Test token batching with a single text that exceeds the limit."""
        # Single text that exceeds token limit
        large_text = "x" * 500000  # ~125k tokens, exceeds 120k limit
        all_texts = [large_text]
        max_tokens_per_batch = 120000
        
        # Mock generate_embeddings
        voyage_client.generate_embeddings = Mock(return_value=[[0.1, 0.2]])
        
        result = voyage_client._generate_embeddings_with_token_batching(
            all_texts, max_tokens_per_batch, "document"
        )
        
        # Should still process the single large text (truncation handled by API)
        assert voyage_client.generate_embeddings.call_count == 1
        assert result == [[0.1, 0.2]]

    def test_generate_embeddings_batch_empty_inputs(self, voyage_client):
        """Test handling of empty inputs."""
        result = voyage_client.generate_embeddings_batch(
            all_texts=[],
            file_boundaries=[],
            max_tokens_per_batch=120000
        )
        
        # Should return empty dict
        assert result == {}

    def test_generate_embeddings_batch_invalid_boundaries(self, voyage_client):
        """Test error handling for invalid file boundaries."""
        all_texts = ["test text"]
        file_boundaries = []  # Empty boundaries with non-empty texts
        
        with pytest.raises(ValueError, match="file_boundaries cannot be empty"):
            voyage_client.generate_embeddings_batch(
                all_texts=all_texts,
                file_boundaries=file_boundaries,
                max_tokens_per_batch=120000
            )

    def test_generate_embeddings_batch_api_error(self, voyage_client):
        """Test handling of API errors during batch processing."""
        all_texts = ["test text"]
        file_boundaries = [("file1.py", 0, 1)]
        
        # Mock estimate_cost
        voyage_client.estimate_cost = Mock(return_value={
            "total_tokens": 1000,
            "total_texts": 1
        })
        
        # Mock API error
        voyage_client.generate_embeddings = Mock(
            side_effect=RuntimeError("API Error: Request failed")
        )
        
        with pytest.raises(RuntimeError, match="Batch embedding generation failed"):
            voyage_client.generate_embeddings_batch(
                all_texts=all_texts,
                file_boundaries=file_boundaries,
                max_tokens_per_batch=120000
            )

    def test_custom_max_tokens_per_batch(self, voyage_client):
        """Test using custom max_tokens_per_batch parameter."""
        all_texts = ["x" * 400, "y" * 400]  # Each ~100 tokens
        file_boundaries = [("file1.py", 0, 1), ("file2.py", 1, 2)]
        
        # Mock estimate_cost to return 200 tokens
        voyage_client.estimate_cost = Mock(return_value={
            "total_tokens": 200,
            "total_texts": 2
        })
        
        # Set custom limit lower than total tokens to force batching
        custom_limit = 150
        
        voyage_client._generate_embeddings_with_token_batching = Mock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )
        
        voyage_client.generate_embeddings_batch(
            all_texts=all_texts,
            file_boundaries=file_boundaries,
            max_tokens_per_batch=custom_limit
        )
        
        # Verify custom limit was passed to batching method
        voyage_client._generate_embeddings_with_token_batching.assert_called_once_with(
            all_texts, custom_limit, "document"
        )
