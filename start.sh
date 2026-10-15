#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo '请先用 Python 3.12 创建 .venv 并安装 requirements.txt。' >&2
    exit 1
fi
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
