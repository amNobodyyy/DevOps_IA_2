#!/usr/bin/env python3
"""
Simple HTTP server for serving the predictions dashboard locally.
This allows the browser to load the JSON files without CORS restrictions.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
PREDICTIONS_DIR = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PREDICTIONS_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for GitHub Pages compatibility
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(PREDICTIONS_DIR)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 Server running at http://localhost:{PORT}")
        print(f"📁 Serving from: {PREDICTIONS_DIR}")
        print("\nOpen http://localhost:8000 in your browser")
        print("Press Ctrl+C to stop the server\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✓ Server stopped")
