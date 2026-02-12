"""Compatibility launcher for local development.

This project now uses a single backend service from backend/app.py.
Run this file only if you want a quick entrypoint from the frontend folder.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app  # noqa: E402


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)