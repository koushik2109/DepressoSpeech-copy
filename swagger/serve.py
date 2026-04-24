"""
Lightweight HTTP server to serve the Swagger UI on port 8080.

Usage:
    python swagger/serve.py              # default port 8080
    python swagger/serve.py --port 9090  # custom port
"""

import argparse
import http.server
import os
import sys
import functools


def parse_args():
    parser = argparse.ArgumentParser(description="Swagger UI Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Serve from the swagger directory
    swagger_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(swagger_dir)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=swagger_dir)

    with http.server.HTTPServer((args.host, args.port), handler) as httpd:
        print(f"\033[0;34m════════════════════════════════════════════════════════════\033[0m")
        print(f"\033[0;34m  Swagger UI – DepressoSpeech API Documentation\033[0m")
        print(f"\033[0;34m════════════════════════════════════════════════════════════\033[0m")
        print(f"\033[0;32m✓ Serving on: http://localhost:{args.port}\033[0m")
        print(f"\033[0;32m✓ Directory:  {swagger_dir}\033[0m")
        print(f"\033[1;33m➜ Open http://localhost:{args.port} in your browser\033[0m")
        print(f"\033[0;34m════════════════════════════════════════════════════════════\033[0m")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\033[0;31m✗ Swagger server stopped.\033[0m")
            sys.exit(0)


if __name__ == "__main__":
    main()
