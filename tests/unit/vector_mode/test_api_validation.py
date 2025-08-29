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
