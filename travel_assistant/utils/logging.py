"""Set up logging, warning, etc."""

import logging
import warnings


class IgnoreOpenAI401Filter(logging.Filter):
    """
    A logging filter that excludes specific OpenAI client error messages.

    Filters out: 'ERROR:openai.agents:[non-fatal] Tracing client error 401'
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Define filter logic."""
        msg = record.getMessage()
        return not (
            record.levelname == "ERROR"
            and record.name == "openai.agents"
            and "[non-fatal] Tracing client error 401" in msg
        )


def set_up_logging():
    """Set up Logging and Warning levels."""
    root_logger = logging.getLogger()
    filter_ = IgnoreOpenAI401Filter()

    if not root_logger.handlers:
        logging.basicConfig()

    for handler in root_logger.handlers:
        handler.addFilter(filter_)

    warnings.filterwarnings("ignore", category=ResourceWarning)
