#!/usr/bin/env python3
"""
Anant Integration Server Launcher
=================================

Simple launcher for the database server on port 8079.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment to avoid config issues
os.environ.setdefault("ANANT_ENV", "development")

if __name__ == "__main__":
    from anant_integration.server.main import run_server
    
    print("🚀 Launching Anant Integration Database Server")
    print("🌐 Server will start on http://localhost:8079")
    print("📖 API docs available at http://localhost:8079/docs")
    print("=" * 50)
    
    try:
        run_server(
            server_type="database",
            host="0.0.0.0",
            port=8079,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)