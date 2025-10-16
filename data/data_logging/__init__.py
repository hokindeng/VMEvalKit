"""Simple dataset version logging."""

from .version_tracker import log_version, get_latest, print_summary

__all__ = ['log_version', 'get_latest', 'print_summary']
__version__ = "1.0.0"