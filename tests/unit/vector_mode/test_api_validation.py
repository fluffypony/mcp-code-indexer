"""
Unit tests for API access validation in TurbopufferClient and VoyageClient.

Tests the validate_api_access methods to ensure they properly detect and
report different types of API access errors with clear error messages.
"""

from unittest.mock import MagicMock, patch
import pytest

from mcp_code_indexer.vector_mode.providers.turbopuffer_client import TurbopufferClient
from mcp_code_indexer.vector_mode.providers.voyage_client import VoyageClient
from mcp_code_indexer.vector_mode.config import VectorConfig


class TestTurbopufferClientValidation:
    """Test TurbopufferClient API access validation."""

    @pytest.fixture
    def mock_turbopuffer_client(self) -> TurbopufferClient:
        """Create a TurbopufferClient with mocked turbopuffer SDK."""
        with patch('turbopuffer.Turbopuffer') as mock_sdk:
            mock_client_instance = MagicMock()
            mock_sdk.return_value = mock_client_instance
            
            client = TurbopufferClient(api_key="test-api-key", region="test-region")
            return client

    def test_validate_api_access_success(self, mock_turbopuffer_client):
        """Test successful API access validation."""
        # Mock successful namespaces call
        mock_turbopuffer_client.client.namespaces.return_value = []
        
        # Should not raise any exception
        mock_turbopuffer_client.validate_api_access()

    def test_validate_api_access_401_unauthorized(self, mock_turbopuffer_client):
        """Test 401 unauthorized error handling."""
        # Mock 401 error
        mock_turbopuffer_client.client.namespaces.side_effect = Exception("401 Unauthorized")
        
        with pytest.raises(RuntimeError, match="Turbopuffer API authentication failed"):
            mock_turbopuffer_client.validate_api_access()

    def test_validate_api_access_403_forbidden(self, mock_turbopuffer_client):
        """Test 403 forbidden error handling."""
        # Mock 403 error
        mock_turbopuffer_client.client.namespaces.side_effect = Exception("403 Forbidden")
        
        with pytest.raises(RuntimeError, match="Turbopuffer API access denied"):
            mock_turbopuffer_client.validate_api_access()

    def test_validate_api_access_429_rate_limit(self, mock_turbopuffer_client):
        """Test 429 rate limit error handling."""
        # Mock rate limit error
        mock_turbopuffer_client.client.namespaces.side_effect = Exception("429 Rate limit exceeded")
        
        with pytest.raises(RuntimeError, match="Turbopuffer API rate limit exceeded"):
            mock_turbopuffer_client.validate_api_access()

    def test_validate_api_access_500_server_error(self, mock_turbopuffer_client):
        """Test 500 server error handling."""
        # Mock server error
        mock_turbopuffer_client.client.namespaces.side_effect = Exception("500 Server error")
        
        with pytest.raises(RuntimeError, match="Turbopuffer service unavailable"):
            mock_turbopuffer_client.validate_api_access()

    def test_validate_api_access_generic_error(self, mock_turbopuffer_client):
        """Test generic error handling."""
        # Mock generic error
        mock_turbopuffer_client.client.namespaces.side_effect = Exception("Network timeout")
        
        with pytest.raises(RuntimeError, match="Turbopuffer API access validation failed"):
            mock_turbopuffer_client.validate_api_access()

    def test_delete_vectors_for_file_success(self, mock_turbopuffer_client):
        """Test successful deletion of vectors for a file."""
        # Mock namespace and query results
        mock_namespace = MagicMock()
        mock_turbopuffer_client.client.namespace.return_value = mock_namespace
        
        # Mock query results with vectors to delete
        mock_row1 = MagicMock()
        mock_row1.id = "test_project_vec_1_abc123"
        mock_row2 = MagicMock()
        mock_row2.id = "test_project_vec_2_def456"
        
        mock_query_result = MagicMock()
        mock_query_result.rows = [mock_row1, mock_row2]
        mock_namespace.query.return_value = mock_query_result
        
        # Mock successful deletion
        mock_delete_result = MagicMock()
        mock_delete_result.rows_affected = 2
        mock_namespace.write.return_value = mock_delete_result
        
        # Test deletion
        result = mock_turbopuffer_client.delete_vectors_for_file("test_namespace", "/test/file.py")
        
        # Verify query was called with correct filter
        mock_namespace.query.assert_called_once_with(
            filters=("file_path", "Eq", "/test/file.py"),
            top_k=1200,
            include_attributes=False
        )
        
        # Verify delete was called with correct IDs
        mock_namespace.write.assert_called_once_with(
            deletes=["test_project_vec_1_abc123", "test_project_vec_2_def456"]
        )
        
        # Verify result
        assert result == {"deleted": 2, "file_path": "/test/file.py"}

    def test_delete_vectors_for_file_no_vectors_found(self, mock_turbopuffer_client):
        """Test deletion when no vectors are found for the file."""
        # Mock namespace and empty query results
        mock_namespace = MagicMock()
        mock_turbopuffer_client.client.namespace.return_value = mock_namespace
        
        # Mock empty query results
        mock_query_result = MagicMock()
        mock_query_result.rows = []
        mock_namespace.query.return_value = mock_query_result
        
        # Test deletion
        result = mock_turbopuffer_client.delete_vectors_for_file("test_namespace", "/test/file.py")
        
        # Verify query was called
        mock_namespace.query.assert_called_once()
        
        # Verify delete was NOT called since no vectors found
        mock_namespace.write.assert_not_called()
        
        # Verify result
        assert result == {"deleted": 0, "file_path": "/test/file.py"}

    def test_delete_vectors_for_file_query_failure(self, mock_turbopuffer_client):
        """Test handling of query failure during deletion."""
        # Mock namespace and query failure
        mock_namespace = MagicMock()
        mock_turbopuffer_client.client.namespace.return_value = mock_namespace
        mock_namespace.query.side_effect = Exception("Query failed")
        
        # Test deletion should raise RuntimeError
        with pytest.raises(RuntimeError, match="File vector deletion failed: Query failed"):
            mock_turbopuffer_client.delete_vectors_for_file("test_namespace", "/test/file.py")

    def test_delete_vectors_for_file_delete_failure(self, mock_turbopuffer_client):
        """Test handling of delete failure."""
        # Mock namespace and query results
        mock_namespace = MagicMock()
        mock_turbopuffer_client.client.namespace.return_value = mock_namespace
        
        # Mock query results with vectors to delete
        mock_row = MagicMock()
        mock_row.id = "test_project_vec_1_abc123"
        mock_query_result = MagicMock()
        mock_query_result.rows = [mock_row]
        mock_namespace.query.return_value = mock_query_result
        
        # Mock delete failure
        mock_namespace.write.side_effect = Exception("Delete failed")
        
        # Test deletion should raise RuntimeError
        with pytest.raises(RuntimeError, match="File vector deletion failed: Vector deletion failed: Delete failed"):
            mock_turbopuffer_client.delete_vectors_for_file("test_namespace", "/test/file.py")


class TestVoyageClientValidation:
    """Test VoyageClient API access validation."""

    @pytest.fixture
    def mock_voyage_client(self) -> VoyageClient:
        """Create a VoyageClient with mocked voyageai SDK."""
        with patch('voyageai.Client') as mock_sdk:
            mock_client_instance = MagicMock()
            mock_sdk.return_value = mock_client_instance
            
            client = VoyageClient(api_key="test-api-key", model="voyage-code-2")
            return client

    def test_validate_api_access_success(self, mock_voyage_client):
        """Test successful API access validation."""
        # Mock successful embed call
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2, 0.3]]
        mock_voyage_client.client.embed.return_value = mock_result
        
        # Should not raise any exception
        mock_voyage_client.validate_api_access()

    def test_validate_api_access_empty_response(self, mock_voyage_client):
        """Test handling of empty API response."""
        # Mock empty response
        mock_result = MagicMock()
        mock_result.embeddings = []
        mock_voyage_client.client.embed.return_value = mock_result
        
        with pytest.raises(RuntimeError, match="Voyage AI API returned empty response"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_401_unauthorized(self, mock_voyage_client):
        """Test 401 unauthorized error handling."""
        # Mock 401 error
        mock_voyage_client.client.embed.side_effect = Exception("401 Invalid API key")
        
        with pytest.raises(RuntimeError, match="Voyage AI API authentication failed"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_403_forbidden(self, mock_voyage_client):
        """Test 403 forbidden error handling."""
        # Mock 403 error
        mock_voyage_client.client.embed.side_effect = Exception("403 Forbidden")
        
        with pytest.raises(RuntimeError, match="Voyage AI API access denied"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_quota_exceeded(self, mock_voyage_client):
        """Test quota exceeded error handling."""
        # Mock quota error
        mock_voyage_client.client.embed.side_effect = Exception("Quota exceeded")
        
        with pytest.raises(RuntimeError, match="Voyage AI API quota exceeded"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_rate_limit(self, mock_voyage_client):
        """Test rate limit error handling."""
        # Mock rate limit error
        mock_voyage_client.client.embed.side_effect = Exception("429 Rate limit exceeded")
        
        with pytest.raises(RuntimeError, match="Voyage AI API rate limit exceeded"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_server_error(self, mock_voyage_client):
        """Test server error handling."""
        # Mock server error
        mock_voyage_client.client.embed.side_effect = Exception("500 Internal server error")
        
        with pytest.raises(RuntimeError, match="Voyage AI service unavailable"):
            mock_voyage_client.validate_api_access()

    def test_validate_api_access_generic_error(self, mock_voyage_client):
        """Test generic error handling."""
        # Mock generic error
        mock_voyage_client.client.embed.side_effect = Exception("Connection failed")
        
        with pytest.raises(RuntimeError, match="Voyage AI API access validation failed"):
            mock_voyage_client.validate_api_access()
