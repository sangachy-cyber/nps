#!/usr/bin/env python3
"""
Simple HTTP server to serve the evaluation report and its images
"""

import os
import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path


def serve_report():
    """Serve the evaluation report using a simple HTTP server"""
    # Define the directory to serve
    report_dir = Path("./data/results/evaluation")
    report_dir = report_dir.resolve()

    # Try different ports if 8000 is already in use
    ports_to_try = [8000, 8001, 8002, 8003, 8004]

    # Create a custom handler that serves files from the report directory
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(report_dir), **kwargs)

    for PORT in ports_to_try:
        try:
            print(f"Serving report from: {report_dir}")

            # Start the server with the custom handler
            with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
                print(f"\nHTTP server started at http://localhost:{PORT}")
                print(
                    f"You can access the report at: http://localhost:{PORT}/comprehensive_evaluation_report_zh.html"
                )
                print("\nPress Ctrl+C to stop the server\n")

                # Open the report in the default browser
                webbrowser.open(
                    f"http://localhost:{PORT}/comprehensive_evaluation_report_zh.html"
                )

                # Serve forever
                httpd.serve_forever()
            break  # Exit loop if server started successfully
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {PORT} is already in use, trying next port...")
                continue
            else:
                raise
        except KeyboardInterrupt:
            print("\nHTTP server stopped.")
            break
    else:
        print(
            "\nFailed to start HTTP server on any of the tried ports. Please try again later."
        )


def main():
    print("Starting HTTP server to serve the evaluation report...")
    print("This will allow you to view the report with all images properly displayed.")
    print("\nThe report will open automatically in your default browser.")
    print(
        "You can also access it manually at: http://localhost:8000/comprehensive_evaluation_report_zh.html"
    )
    print("\nTo stop the server, press Ctrl+C in this terminal.")
    print("=" * 70)

    # Start the server
    serve_report()


if __name__ == "__main__":
    main()
