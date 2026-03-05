#!/usr/bin/env python3
"""
便捷启动脚本：在项目根目录执行 `uv run python run.py` 即可启动 Amadeus 服务。
"""

from amadeus.main import run

if __name__ == "__main__":
    run()
