import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich import box
from datetime import datetime

# ============================= KONFIGURÁCIA =====================================
# Tu môžeš meniť všetky parametre podľa potreby

TICKERS = ["AAPL", "MSFT", "COST", "NVDA", "META", "GOOGL", "AMZN", "TSLA"]  # Môžeš pridať koľko chceš
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
LOOKBACK_PERIODS = [10, 20, 30, 40, 50, 60]  # Konfigurácie lookback periód
SHOW_PLOTS = True  # True = zobraz grafy, False = bez grafov
TOP_N = 5  # Koľko top konfigurácií zobraziť

# ================================================================================

console = Console()

# Hackerský banner
banner = """
================================================================================
  ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
  ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
     ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
     ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
     ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                           
         D O N C H I A N   B R E A K O U T   S Y S T E M 
                  [v1.0 - Matrix Edition]                 
================================================================================
"""

console.print(f"[bold green]{banner}[/bold green]")
console.print(Panel.fit(
    f"[bold cyan]INITIALIZING BACKTEST ENGINE[/bold cyan]\n"
    f"[green]> Tickers: {', '.join(TICKERS)}[/green]\n"
    f"[green]> Period: {START_DATE} to {END_DATE}[/green]\n"
    f"[green]> Lookback configs: {LOOKBACK_PERIODS}[/green]\n"
    f"[yellow]> Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/yellow]",
    border_style="bright_green",
    box=box.DOUBLE
))

def rolling_max_min(prices, window):
    """Vypočíta rolling maximum a minimum pre dané okno"""
    number_of_prices = len(prices)
    rolling_max = np.full(number_of_prices, np.nan)
    rolling_min = np.full(number_of_prices, np.nan)

    for i in range(window - 1, number_of_prices):
        rolling_max[i] = np.max(prices[i - window + 1:i + 1])
        rolling_min[i] = np.min(prices[i - window + 1:i + 1])

    return rolling_max, rolling_min

def backtest(prices, returns, lookback):
    """Vykoná backtest Donchian breakout stratégie"""
    upper_channel, lower_channel = rolling_max_min(prices, lookback)       

    signals = np.zeros(len(prices))
    position = 0

    for i in range(lookback, len(prices)):
        if position == 0:  
            if prices[i] > upper_channel[i - 1]:
                position = 1
        elif position == 1:
            if prices[i] < lower_channel[i - 1]:
                position = 0

        signals[i] = position

    strategy_returns = returns * signals
    strategy_returns[:lookback] = 0

    total_return = np.prod(1 + strategy_returns) - 1
    sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)

    cumulative = np.cumprod(1 + strategy_returns)
    runningmax = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - runningmax) / runningmax
    max_drawdown = np.min(drawdown)

    num_trades = np.sum(np.abs(np.diff(signals)) > 0)

    return total_return, sharpe, max_drawdown, num_trades, cumulative

all_results = []
failed_tickers = []
best_equity_curves = {}  # Pre uloženie equity kriviek top konfigurácií

console.print("\n[bold green]>>> SCANNING MARKETS...[/bold green]\n")

for ticker in track(TICKERS, description="[green]Processing stocks..."):
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

        if data.empty or len(data) < 100:
            failed_tickers.append(ticker)
            console.print(f"[red]X {ticker} - Insufficient data[/red]")
            continue

        if isinstance(data["Close"], pd.DataFrame):
            prices = data["Close"].iloc[:,0].values
        else:
            prices = data["Close"].values

        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])

        for lookback in LOOKBACK_PERIODS:
            total_return, sharpe, max_drawdown, num_trades, cumulative = backtest(prices, returns, lookback)

            config_key = f"{ticker}_{lookback}"
            best_equity_curves[config_key] = cumulative

            all_results.append({
                'ticker': ticker,
                'lookback': lookback,
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100,
                'num_trades': num_trades,
                'buy_hold_return': (prices[-1] / prices[0] - 1) * 100,
                'config_key': config_key
            })
        
        console.print(f"[green]OK {ticker} - Complete[/green]")

    except Exception as e:
        failed_tickers.append(ticker)
        console.print(f"[red]X {ticker} - Error: {str(e)}[/red]")
        continue 

results_df = pd.DataFrame(all_results)

# Hľadanie najlepších konfigurácií
console.print("\n" + "="*80)
console.print("[bold green]>>> ANALYZING OPTIMAL CONFIGURATIONS...[/bold green]\n")

best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
best_sharpe = results_df.loc[best_sharpe_idx]

best_return_idx = results_df['total_return'].idxmax()
best_return = results_df.loc[best_return_idx]

top_configs = results_df.nlargest(TOP_N, 'sharpe_ratio')
top_returns = results_df.nlargest(TOP_N, 'total_return')

# TABUĽKA: Najlepšia konfigurácia
table_best = Table(title="[bold green]BEST CONFIGURATION BY SHARPE RATIO[/bold green]", 
                   box=box.DOUBLE_EDGE, 
                   border_style="bright_green",
                   show_header=True,
                   header_style="bold cyan",
                   expand=True)

table_best.add_column("Metric", style="cyan", width=25)
table_best.add_column("Value", style="green bold", width=30)

table_best.add_row("Ticker", f"[yellow]{best_sharpe['ticker']}[/yellow]")
table_best.add_row("Lookback Period", f"[yellow]{int(best_sharpe['lookback'])} days[/yellow]")
table_best.add_row("Total Return", f"[green]{best_sharpe['total_return']:.2f}%[/green]")
table_best.add_row("Sharpe Ratio", f"[green bold]{best_sharpe['sharpe_ratio']:.3f}[/green bold]")
table_best.add_row("Max Drawdown", f"[red]{best_sharpe['max_drawdown']:.2f}%[/red]")
table_best.add_row("Number of Trades", f"[yellow]{int(best_sharpe['num_trades'])}[/yellow]")
table_best.add_row("Buy & Hold Return", f"[blue]{best_sharpe['buy_hold_return']:.2f}%[/blue]")
table_best.add_row("Strategy vs B&H", f"[magenta]{best_sharpe['total_return'] - best_sharpe['buy_hold_return']:.2f}%[/magenta]")

console.print(table_best)

# TABUĽKA: TOP N podľa Sharpe
console.print("\n")
table_top = Table(title=f"[bold cyan]TOP {TOP_N} CONFIGURATIONS BY SHARPE RATIO[/bold cyan]",
                  box=box.HEAVY,
                  border_style="cyan",
                  show_header=True,
                  header_style="bold green",
                  expand=True)

table_top.add_column("Rank", justify="center", style="yellow bold", width=6)
table_top.add_column("Ticker", style="cyan", width=8)
table_top.add_column("Lookback", justify="right", style="magenta", width=10)
table_top.add_column("Return %", justify="right", style="green", width=12)
table_top.add_column("Sharpe", justify="right", style="green bold", width=10)
table_top.add_column("Drawdown %", justify="right", style="red", width=12)
table_top.add_column("Trades", justify="right", style="yellow", width=8)

for idx, (_, row) in enumerate(top_configs.iterrows(), 1):
    table_top.add_row(
        f"#{idx}",
        row['ticker'],
        f"{int(row['lookback'])}d",
        f"{row['total_return']:.2f}",
        f"{row['sharpe_ratio']:.3f}",
        f"{row['max_drawdown']:.2f}",
        f"{int(row['num_trades'])}"
    )

console.print(table_top)

# TABUĽKA: TOP N podľa Return
console.print("\n")
table_top_ret = Table(title=f"[bold yellow]TOP {TOP_N} CONFIGURATIONS BY TOTAL RETURN[/bold yellow]",
                      box=box.HEAVY,
                      border_style="yellow",
                      show_header=True,
                      header_style="bold green",
                      expand=True)

table_top_ret.add_column("Rank", justify="center", style="yellow bold", width=6)
table_top_ret.add_column("Ticker", style="cyan", width=8)
table_top_ret.add_column("Lookback", justify="right", style="magenta", width=10)
table_top_ret.add_column("Return %", justify="right", style="green bold", width=12)
table_top_ret.add_column("Sharpe", justify="right", style="green", width=10)
table_top_ret.add_column("Drawdown %", justify="right", style="red", width=12)
table_top_ret.add_column("Trades", justify="right", style="yellow", width=8)

for idx, (_, row) in enumerate(top_returns.iterrows(), 1):
    table_top_ret.add_row(
        f"#{idx}",
        row['ticker'],
        f"{int(row['lookback'])}d",
        f"{row['total_return']:.2f}",
        f"{row['sharpe_ratio']:.3f}",
        f"{row['max_drawdown']:.2f}",
        f"{int(row['num_trades'])}"
    )

console.print(table_top_ret)

# TABUĽKA: Porovnanie tickerov
console.print("\n")
table_comparison = Table(title="[bold magenta]BEST CONFIGURATION PER TICKER (by Sharpe)[/bold magenta]",
                         box=box.DOUBLE,
                         border_style="magenta",
                         show_header=True,
                         header_style="bold cyan",
                         expand=True)

table_comparison.add_column("Ticker", style="cyan bold", width=10)
table_comparison.add_column("Best Lookback", justify="right", style="magenta", width=14)
table_comparison.add_column("Return %", justify="right", style="green", width=12)
table_comparison.add_column("Sharpe", justify="right", style="green bold", width=10)
table_comparison.add_column("Drawdown %", justify="right", style="red", width=13)
table_comparison.add_column("Trades", justify="right", style="yellow", width=8)
table_comparison.add_column("B&H %", justify="right", style="blue", width=10)
table_comparison.add_column("Edge", justify="right", style="magenta", width=10)

for ticker in TICKERS:
    ticker_data = results_df[results_df['ticker'] == ticker]
    if not ticker_data.empty:
        best = ticker_data.loc[ticker_data['sharpe_ratio'].idxmax()]
        edge = best['total_return'] - best['buy_hold_return']
        edge_color = "green" if edge > 0 else "red"
        
        table_comparison.add_row(
            best['ticker'],
            f"{int(best['lookback'])}d",
            f"{best['total_return']:.2f}",
            f"{best['sharpe_ratio']:.3f}",
            f"{best['max_drawdown']:.2f}",
            f"{int(best['num_trades'])}",
            f"{best['buy_hold_return']:.2f}",
            f"[{edge_color}]{edge:+.2f}[/{edge_color}]"
        )

console.print(table_comparison)

# Štatistiky
console.print("\n" + "="*80)
console.print("[bold green]>>> PORTFOLIO STATISTICS[/bold green]\n")

table_stats = Table(title="[bold cyan]SUMMARY STATISTICS[/bold cyan]",
                    box=box.DOUBLE,
                    border_style="bright_green",
                    show_header=True,
                    header_style="bold cyan",
                    expand=True)

table_stats.add_column("Metric", style="cyan", width=30)
table_stats.add_column("Value", style="green bold", width=25)

table_stats.add_row("Total Configurations Tested", f"[yellow]{len(results_df)}[/yellow]")
table_stats.add_row("Average Sharpe Ratio", f"[green]{results_df['sharpe_ratio'].mean():.3f}[/green]")
table_stats.add_row("Average Total Return", f"[green]{results_df['total_return'].mean():.2f}%[/green]")
table_stats.add_row("Average Max Drawdown", f"[red]{results_df['max_drawdown'].mean():.2f}%[/red]")
table_stats.add_row("Average Trades per Config", f"[yellow]{results_df['num_trades'].mean():.1f}[/yellow]")
table_stats.add_row("Best Performing Ticker", f"[yellow]{best_return['ticker']}[/yellow] [green]({best_return['total_return']:.2f}%)[/green]")
table_stats.add_row("Most Consistent (Sharpe)", f"[yellow]{best_sharpe['ticker']}[/yellow] [green]({best_sharpe['sharpe_ratio']:.3f})[/green]")

console.print(table_stats)

if failed_tickers:
    console.print(f"\n[yellow]Warning: {len(failed_tickers)} ticker(s) failed: {', '.join(failed_tickers)}[/yellow]")

console.print("\n" + "="*80)
console.print("[bold green]>>> GENERATING VISUALIZATIONS...[/bold green]\n")

# ========================= MATPLOTLIB GRAFY =========================

if SHOW_PLOTS:
    # Nastavenie štýlu - čierne pozadie, zelené grafy
    plt.style.use('dark_background')
    
    # Vytvorenie 2x2 grid grafov s väčším priestorom
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Graf 1: TOP konfigurácie - Equity curves (ľavý horný)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('#0a0a0a')
    
    colors = ['#00ff41', '#00ff88', '#00ffcc', '#66ff66', '#88ff88']
    
    for idx, (_, row) in enumerate(top_configs.iterrows()):
        if idx < len(colors):
            equity = best_equity_curves[row['config_key']]
            label = f"{row['ticker']} ({int(row['lookback'])}d) SR:{row['sharpe_ratio']:.2f}"
            ax1.plot(equity, color=colors[idx], linewidth=2.5, label=label, alpha=0.9)
    
    ax1.set_title(f'TOP {TOP_N} STRATEGIES - EQUITY CURVES', 
                  fontsize=12, color='#00ff41', fontweight='bold', pad=12)
    ax1.set_xlabel('Trading Days', fontsize=10, color='#ffffff', labelpad=8)
    ax1.set_ylabel('Cumulative Return', fontsize=10, color='#ffffff', labelpad=8)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.95, 
              facecolor='#0a0a0a', edgecolor='#00ff41', labelcolor='#ffffff')
    ax1.grid(True, alpha=0.15, color='#00ff41', linestyle='--', linewidth=0.5)
    ax1.tick_params(colors='#ffffff', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#00ff41')
    
    # Graf 2: Return vs Sharpe scatter (pravý horný)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('#0a0a0a')
    
    scatter_colors = ['#00ff41', '#00ff88', '#00ffcc', '#66ff66', '#88ff88', 
                     '#99ff99', '#aaffaa', '#bbffbb', '#ccffcc', '#ddffdd']
    
    for idx, ticker in enumerate(TICKERS):
        ticker_data = results_df[results_df['ticker'] == ticker]
        if not ticker_data.empty:
            color = scatter_colors[idx % len(scatter_colors)]
            ax2.scatter(ticker_data['sharpe_ratio'], ticker_data['total_return'], 
                       s=100, alpha=0.7, label=ticker, color=color, 
                       edgecolors='#ffffff', linewidth=0.8)
    
    ax2.set_title('RETURN vs SHARPE RATIO', 
                  fontsize=12, color='#00ff41', fontweight='bold', pad=12)
    ax2.set_xlabel('Sharpe Ratio', fontsize=10, color='#ffffff', labelpad=8)
    ax2.set_ylabel('Total Return (%)', fontsize=10, color='#ffffff', labelpad=8)
    ax2.legend(loc='best', fontsize=8, framealpha=0.95, ncol=2,
              facecolor='#0a0a0a', edgecolor='#00ff41', labelcolor='#ffffff')
    ax2.grid(True, alpha=0.15, color='#00ff41', linestyle='--', linewidth=0.5)
    ax2.tick_params(colors='#ffffff', labelsize=8)
    ax2.axhline(y=0, color='#ff4444', linestyle='--', alpha=0.6, linewidth=1.5)
    ax2.axvline(x=0, color='#ff4444', linestyle='--', alpha=0.6, linewidth=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#00ff41')
    
    # Graf 3: Heatmapa Return (ľavý dolný)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('#0a0a0a')
    
    pivot_return = results_df.pivot(index='ticker', columns='lookback', values='total_return')
    
    # Použitie vlastnej zelenej colormapy
    from matplotlib.colors import LinearSegmentedColormap
    colors_map = ['#000000', '#003300', '#006600', '#009900', '#00cc00', '#00ff00']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_green', colors_map, N=n_bins)
    
    im = ax3.imshow(pivot_return.values, cmap=cmap, aspect='auto', interpolation='nearest')
    
    ax3.set_xticks(np.arange(len(pivot_return.columns)))
    ax3.set_yticks(np.arange(len(pivot_return.index)))
    ax3.set_xticklabels([f"{int(x)}d" for x in pivot_return.columns], 
                        color='#ffffff', fontsize=9)
    ax3.set_yticklabels(pivot_return.index, color='#ffffff', fontsize=9)
    
    ax3.set_xlabel('Lookback Period', fontsize=10, color='#ffffff', labelpad=8)
    ax3.set_ylabel('Ticker', fontsize=10, color='#ffffff', labelpad=8)
    ax3.set_title('TOTAL RETURN HEATMAP (%)', 
                  fontsize=12, color='#00ff41', fontweight='bold', pad=12)
    
    # Pridanie čísel do heatmapy
    for i in range(len(pivot_return.index)):
        for j in range(len(pivot_return.columns)):
            value = pivot_return.values[i, j]
            text_color = '#000000' if value > pivot_return.values.mean() else '#ffffff'
            ax3.text(j, i, f'{value:.0f}', ha='center', va='center', 
                    color=text_color, fontsize=8, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, pad=0.02)
    cbar.set_label('Return (%)', rotation=270, labelpad=18, color='#ffffff', fontsize=9)
    cbar.ax.tick_params(colors='#ffffff', labelsize=8)
    cbar.outline.set_edgecolor('#00ff41')
    
    # Graf 4: Bar chart - Najlepšie konfigurácie podľa tickera (pravý dolný)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('#0a0a0a')
    
    best_per_ticker = []
    tickers_list = []
    for ticker in TICKERS:
        ticker_data = results_df[results_df['ticker'] == ticker]
        if not ticker_data.empty:
            best = ticker_data.loc[ticker_data['sharpe_ratio'].idxmax()]
            best_per_ticker.append(best['total_return'])
            tickers_list.append(ticker)
    
    bars = ax4.bar(tickers_list, best_per_ticker, color='#00ff41', 
                   alpha=0.8, edgecolor='#ffffff', linewidth=1.5, width=0.6)
    
    # Pridanie hodnôt nad stĺpce s odstupom
    for bar in bars:
        height = bar.get_height()
        offset = 5 if height > 0 else -15
        ax4.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                color='#ffffff', fontsize=9, fontweight='bold')
    
    ax4.set_title('BEST RETURN PER TICKER', 
                  fontsize=12, color='#00ff41', fontweight='bold', pad=12)
    ax4.set_xlabel('Ticker', fontsize=10, color='#ffffff', labelpad=8)
    ax4.set_ylabel('Total Return (%)', fontsize=10, color='#ffffff', labelpad=8)
    ax4.grid(True, alpha=0.15, color='#00ff41', linestyle='--', 
            linewidth=0.5, axis='y')
    ax4.tick_params(colors='#ffffff', labelsize=9)
    ax4.axhline(y=0, color='#ff4444', linestyle='--', alpha=0.6, linewidth=1.5)
    for spine in ax4.spines.values():
        spine.set_color('#00ff41')
    
    # Nastavenie y-limitov s marginmi pre bar chart
    if best_per_ticker:
        y_min = min(best_per_ticker)
        y_max = max(best_per_ticker)
        margin = (y_max - y_min) * 0.15
        ax4.set_ylim(y_min - margin, y_max + margin)
    
    plt.suptitle('DONCHIAN BREAKOUT STRATEGY - PERFORMANCE ANALYSIS', 
                fontsize=15, color='#00ff41', fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.985], h_pad=3.0, w_pad=2.5)
    plt.show()
    
    console.print("[green]OK Visualizations generated successfully[/green]")

console.print("\n" + "="*80)
console.print("[bold green]>>> BACKTEST COMPLETE[/bold green]")
console.print("[dim]System ready for deployment...[/dim]\n")