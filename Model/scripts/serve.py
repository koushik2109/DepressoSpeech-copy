"""
[LAYER_START] Session 9: API Server Entry Point
Launches the FastAPI application with uvicorn.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080
    python scripts/serve.py --config configs/inference_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn

from src.api.app import create_app
from src.utils import setup_logging as _setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="DepressoSpeech API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="Path to SSL private key file for HTTPS (SEC-5)",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="Path to SSL certificate file for HTTPS (SEC-5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_path = _setup_logging(
        task_name="api_server",
        log_dir="logs",
        console_level=args.log_level.upper(),
        file_level="DEBUG",
    )
    logger = logging.getLogger(__name__)

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("DepressoSpeech API Server")
    logger.info("=" * 60)
    logger.info(f"Config:  {args.config}")
    logger.info(f"Host:    {args.host}")
    logger.info(f"Port:    {args.port}")
    logger.info(f"Reload:  {args.reload}")
    logger.info(f"TLS:     {'enabled' if args.ssl_certfile else 'disabled'}")
    logger.info(f"Log:     {log_path}")
    logger.info("=" * 60)

    if not args.ssl_certfile:
        logger.warning(
            "[SEC-5] TLS is disabled. Patient audio and PHQ-8 scores will be transmitted "
            "in plaintext. Set --ssl-certfile and --ssl-keyfile for HIPAA compliance."
        )

    app = create_app(config_path=args.config)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
