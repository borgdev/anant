#!/usr/bin/env python3
"""
Anant Graph API Startup Script
=============================

Entry point for running the Anant Graph API within the Ray cluster.
"""

import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1  # Single worker for Ray deployment
    )