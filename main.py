"""
ETH Trend AI — Entry point.

Usage:
    python main.py              # single analysis, default 90 days of data
    python main.py --days 180   # use 180 days of data
    python main.py --watch 60   # refresh every 60 seconds
    python main.py --raw        # print raw scores to stdout (no rich UI)
"""

import argparse
import time
import sys

from rich.console import Console

console = Console()


def run_analysis(days: int, raw: bool = False):
    from src.fetcher import fetch_ohlc, fetch_market_data
    from src.indicators import add_all_indicators
    from src.trend_detector import analyze
    from src.dashboard import render_full_dashboard

    console.print("[dim]  Fetching ETH market data...[/]")

    try:
        ohlc_df = fetch_ohlc(days=days)
        market = fetch_market_data()
    except ConnectionError as e:
        console.print(f"[bold red]Connection error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {e}")
        sys.exit(1)

    console.print(f"[dim]  Processing {len(ohlc_df)} candles...[/]")
    df_indicators = add_all_indicators(ohlc_df)

    if df_indicators.empty:
        console.print("[bold red]Not enough data after computing indicators. Try more days.[/]")
        sys.exit(1)

    result = analyze(df_indicators)

    if raw:
        # Simple text output for scripting / integration
        print(f"TREND={result.trend}")
        print(f"REVERSAL_RISK={result.reversal_risk}")
        print(f"CONFIDENCE={result.confidence:.1f}")
        print(f"SCORE={result.total_score:.2f}")
        print(f"RECOMMENDATION={result.recommendation}")
        for r in result.reversal_reasons:
            print(f"REVERSAL_WARNING={r}")
    else:
        render_full_dashboard(market, result)


def main():
    parser = argparse.ArgumentParser(
        description="ETH Trend AI — Detect market trend and reversal signals for Ethereum"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical OHLC data to fetch (default: 90, max: 365)",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Auto-refresh interval in seconds (0 = run once)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output plain key=value text instead of rich dashboard",
    )
    args = parser.parse_args()

    if args.days < 10:
        console.print("[bold red]--days must be at least 10[/]")
        sys.exit(1)
    if args.days > 365:
        console.print("[yellow]Warning: CoinGecko free tier supports max 365 days.[/]")

    if args.watch > 0:
        console.print(f"[bold blue]Watch mode:[/] refreshing every {args.watch}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                run_analysis(args.days, raw=args.raw)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/]")
    else:
        run_analysis(args.days, raw=args.raw)


if __name__ == "__main__":
    main()
