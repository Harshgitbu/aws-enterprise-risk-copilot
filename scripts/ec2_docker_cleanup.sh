#!/usr/bin/env bash
set -euo pipefail

echo "== Docker disk usage before cleanup =="
docker system df || true

echo ""
echo "Stopping exited containers and pruning unused artifacts..."
docker container prune -f
docker image prune -af
docker builder prune -af
docker volume prune -f

echo ""
echo "== Docker disk usage after cleanup =="
docker system df || true

echo ""
echo "Done. If you still need space, also clean host logs:"
echo "  sudo journalctl --vacuum-time=7d"
