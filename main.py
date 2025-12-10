from src.utils.logging_config import setup_logging, CustomLogger
from typing import cast


def main():
    logger = setup_logging(full_color=True, include_function=True)
    logger = cast(CustomLogger, logger)  # Type hinting
    logger.success("Logging configuration test completed.")


if __name__ == "__main__":
    main()
