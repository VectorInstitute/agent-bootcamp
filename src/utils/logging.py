"""Set up logging, warning, etc."""

import warnings


def set_up_logging():
    """Set up Logging and Warning levels."""
    warnings.filterwarnings("ignore", category=ResourceWarning)
