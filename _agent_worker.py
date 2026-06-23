#!/usr/bin/env python3
"""獨立子程序 worker — 執行 TradingAgents 分析，避免 macOS spawn 問題。

用法 (由 agent_gate.py 透過 subprocess 呼叫):
    echo '{"ticker":"AAPL","trade_date":"2026-06-23","config":{...}}' | python3 _agent_worker.py

輸入: JSON (stdin) — ticker, trade_date, config
輸出: JSON (stdout) — decision_text, signal
"""

import json
import os
import sys


def main():
    data = json.loads(sys.stdin.read())
    ticker = data["ticker"]
    trade_date = data["trade_date"]
    config = data["config"]

    # 確保 API key
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    ta = TradingAgentsGraph(config=config)
    final_state, signal = ta.propagate(ticker, trade_date)

    decision_text = final_state.get("final_trade_decision", "")
    json.dump({"decision_text": decision_text, "signal": signal}, sys.stdout)


if __name__ == "__main__":
    main()
