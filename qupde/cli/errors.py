class ParseError(Exception):
    """Raised when user-provided equations cannot be parsed or validated."""


class QuadratizationError(Exception):
    """Raised when quadratization fails or returns empty."""
