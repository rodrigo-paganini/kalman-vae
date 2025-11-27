import logging
from pathlib import Path


def setup_logging(log_file: str = None, level=logging.INFO):
    """
    Initialize Python logging and patch tqdm so it doesn't break the log output.
    Call once at program startup.
    """

    # Prevent double initialization
    if logging.getLogger().handlers:
        return

    handlers = []

    # Console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    handlers.append(console)

    # Optional file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        f = logging.FileHandler(log_file, mode="w")
        f.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        handlers.append(f)

    logging.basicConfig(level=level, handlers=handlers)


