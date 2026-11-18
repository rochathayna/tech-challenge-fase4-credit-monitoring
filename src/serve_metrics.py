"""Expõe reports/metrics.prom em http://localhost:8000/metrics para scrape do Prometheus."""
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import REPORTS_DIR


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = (REPORTS_DIR / "metrics.prom").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("Prometheus metrics em http://localhost:8000/metrics")
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
