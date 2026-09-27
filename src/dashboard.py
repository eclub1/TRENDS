"""
Rich terminal dashboard for ETH trend analysis output.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box
from datetime import datetime

from .trend_detector import TrendResult

console = Console()


# ── Color helpers ────────────────────────────────────────────────────────────

TREND_STYLES = {
    "STRONG_UPTREND":   ("bold green",  "🚀"),
    "UPTREND":          ("green",       "📈"),
    "NEUTRAL":          ("yellow",      "➡️ "),
    "DOWNTREND":        ("red",         "📉"),
    "STRONG_DOWNTREND": ("bold red",    "🔻"),
}

RISK_STYLES = {
    "LOW":      "bold green",
    "MEDIUM":   "bold yellow",
    "HIGH":     "bold red",
    "CRITICAL": "bold white on red",
}


def _trend_bar(score: float) -> str:
    """Visual -10 to +10 score bar."""
    clamped = max(-10, min(10, score))
    filled = int(abs(clamped))
    if clamped > 0:
        return "[green]" + "█" * filled + "░" * (10 - filled) + "[/]" + f" +{score:.2f}"
    elif clamped < 0:
        return "[red]" + "█" * filled + "░" * (10 - filled) + "[/]" + f" {score:.2f}"
    else:
        return "░" * 10 + "  0.00"


def _signal_score_bar(score: float) -> str:
    """Mini score indicator for individual signals."""
    if score > 0:
        return f"[green]+{score:.1f}[/]"
    elif score < 0:
        return f"[red]{score:.1f}[/]"
    else:
        return "[yellow] 0.0[/]"


def render_header(market: dict):
    price = market.get("price", 0)
    change_24h = market.get("change_24h", 0)
    change_7d = market.get("change_7d", 0)
    volume = market.get("volume_24h", 0)
    high = market.get("high_24h", 0)
    low = market.get("low_24h", 0)
    updated = market.get("last_updated", "")

    change_24h_color = "green" if change_24h >= 0 else "red"
    change_7d_color = "green" if change_7d >= 0 else "red"
    sign_24h = "+" if change_24h >= 0 else ""
    sign_7d = "+" if change_7d >= 0 else ""

    header = Text()
    header.append("  ETH / USD  ", style="bold white on blue")
    header.append(f"  ${price:,.2f}", style="bold cyan")
    header.append(f"  24h: ", style="dim")
    header.append(f"{sign_24h}{change_24h:.2f}%", style=change_24h_color)
    header.append(f"  7d: ", style="dim")
    header.append(f"{sign_7d}{change_7d:.2f}%", style=change_7d_color)

    stats = (
        f"[dim]Vol 24h:[/] [white]${volume/1e9:.2f}B[/]  "
        f"[dim]High:[/] [white]${high:,.0f}[/]  "
        f"[dim]Low:[/] [white]${low:,.0f}[/]  "
        f"[dim]Updated:[/] [white]{updated[:19]}[/]"
    )

    console.print()
    console.print(Panel(header, subtitle=stats, border_style="blue", padding=(0, 2)))


def render_trend_summary(result: TrendResult):
    style, emoji = TREND_STYLES.get(result.trend, ("white", ""))
    risk_style = RISK_STYLES.get(result.reversal_risk, "white")

    trend_text = Text()
    trend_text.append(f"{emoji}  {result.trend.replace('_', ' ')}", style=style)

    score_bar = _trend_bar(result.total_score)

    summary_table = Table.grid(expand=True, padding=(0, 2))
    summary_table.add_column(ratio=1)
    summary_table.add_column(ratio=1)
    summary_table.add_column(ratio=1)

    summary_table.add_row(
        f"[bold]Trend[/]\n{trend_text}",
        f"[bold]Reversal Risk[/]\n[{risk_style}]{result.reversal_risk}[/]",
        f"[bold]Confidence[/]\n[cyan]{result.confidence:.0f}%[/]",
    )

    console.print(Rule("[bold]TREND ANALYSIS[/]", style="blue"))
    console.print(Panel(summary_table, border_style=style.split()[-1], padding=(1, 2)))
    console.print(f"  Score: {score_bar}")
    console.print()


def render_signals(result: TrendResult):
    table = Table(
        title="Signal Breakdown",
        box=box.ROUNDED,
        border_style="dim",
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("Indicator", style="bold", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Score", justify="center")
    table.add_column("Interpretation")

    for sig in result.signals:
        score_str = _signal_score_bar(sig.score)
        # Format value nicely
        if isinstance(sig.value, float):
            val_str = f"{sig.value:.2f}"
        else:
            val_str = str(sig.value)

        interp_style = "green" if sig.score > 0 else ("red" if sig.score < 0 else "yellow")
        table.add_row(
            sig.name,
            val_str,
            score_str,
            f"[{interp_style}]{sig.interpretation}[/]",
        )

    console.print(table)
    console.print()


def render_reversal_warnings(result: TrendResult):
    if not result.reversal_reasons:
        console.print("[dim]  No reversal warnings detected.[/]")
        console.print()
        return

    risk_style = RISK_STYLES.get(result.reversal_risk, "white")
    console.print(Rule(f"[{risk_style}]REVERSAL SIGNALS — {result.reversal_risk} RISK[/]", style="red"))

    for reason in result.reversal_reasons:
        console.print(f"  [red]⚠[/]  {reason}")
    console.print()


def render_recommendation(result: TrendResult):
    trend_style, _ = TREND_STYLES.get(result.trend, ("white", ""))
    console.print(
        Panel(
            f"[bold]{result.recommendation}[/]",
            title="[bold]AI Recommendation[/]",
            border_style=trend_style.split()[-1],
            padding=(1, 3),
        )
    )
    console.print()


def render_disclaimer():
    console.print(
        "[dim]⚠  This tool is for informational and educational purposes only. "
        "Nothing here constitutes financial advice. Always do your own research.[/dim]\n"
    )


def render_full_dashboard(market: dict, result: TrendResult):
    console.clear()
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    console.print(f"\n[bold blue]ETH Trend AI[/]  [dim]·  {now}[/]")
    render_header(market)
    render_trend_summary(result)
    render_signals(result)
    render_reversal_warnings(result)
    render_recommendation(result)
    render_disclaimer()
