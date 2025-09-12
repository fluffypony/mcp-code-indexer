"""
Constants for Voyage AI providers.

Contains model names, dimensions, and other constants used across
Voyage AI provider implementations.
"""


class VoyageModel:
    """Voyage AI model names."""

    VOYAGE_CODE_2 = "voyage-code-2"
    VOYAGE_2 = "voyage-2"
    VOYAGE_LARGE_2 = "voyage-large-2"
    VOYAGE_3 = "voyage-3"


# Model dimensions mapping
MODEL_DIMENSIONS = {
    VoyageModel.VOYAGE_CODE_2: 1536,
    VoyageModel.VOYAGE_2: 1024,
    VoyageModel.VOYAGE_LARGE_2: 1536,
    VoyageModel.VOYAGE_3: 1024,
}
