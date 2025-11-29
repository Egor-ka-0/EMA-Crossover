import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

ticker = "BTC-USD"
start_date = "2025-01-01"
end_date = "2025-11-22"
fast_ema_period = 1
slow_ema_period = 2
initial_capital = 10000

def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_price_data(raw_data: pd.DataFrame):
    data = raw_data.copy()
    data = data[['Close']]
    data.dropna(inplace=True)
    data.rename(columns={'Close': 'price'}, inplace=True)
    return data

def add_ema(price_data: pd.DataFrame,
            fast_period: int,
            slow_period: int):
    data = price_data.copy()
    data['ema_fast'] = data['price'].ewm(span=fast_period, adjust=False).mean()
    data['ema_slow'] = data['price'].ewm(span=slow_period, adjust=False).mean()
    return data

def add_position(data_with_ema: pd.DataFrame):
    data = data_with_ema.copy()
    mask = data['ema_fast'] > data['ema_slow']
    data['position'] = mask.astype(int)
    return data

def run_backtest(data_with_position: pd.DataFrame,
                 initial_capital: float):
    data = data_with_position.copy()
    data['asset_return'] = data['price'].pct_change()
    data['strategy_return'] = data['asset_return'] * data['position']
    data['equity_curve'] = (1 + data['strategy_return']).cumprod() * initial_capital
    return data

def compute_metrics(result: pd.DataFrame,
                    initial_capital: float) -> dict:
    data = result.copy()
    final_capital = data['equity_curve'].iloc[-1]
    total_return = final_capital / initial_capital - 1

    running_max = data['equity_curve'].cummax()
    data['drawdown'] = data['equity_curve'] / running_max - 1
    max_drawdown = data['drawdown'].min()
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }
def plot_price_and_ema(data: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    line_price, = ax.plot(data.index, data['price'], label='Price')
    line_fast, = ax.plot(data.index, data['ema_fast'], label='EMA fast')
    line_slow, = ax.plot(data.index, data['ema_slow'], label='EMA slow')
    ax.set_title(f"{ticker} price and EMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc='upper left')
    rax = plt.axes([0.84, 0.84, 0.15, 0.15])
    labels = ['Price', 'EMA fast', 'EMA slow']
    visibility = [True, True, True]
    check = CheckButtons(rax, labels, visibility)
    def toggle_line(label):
        if label == 'Price':
            line = line_price
        elif label == 'EMA fast':
            line = line_fast
        else:  # 'EMA slow'
            line = line_slow
        line.set_visible(not line.get_visible())
        plt.draw()
    check.on_clicked(toggle_line)
    plt.show()

def plot_equity_curve(data: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['equity_curve'], label='Equity curve')
    plt.title("Equity curve")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True)
    plt.show()

raw_data = load_data(ticker, start_date, end_date)
price_data = prepare_price_data(raw_data)
data_with_ema = add_ema(price_data, fast_ema_period, slow_ema_period)
data_with_position = add_position(data_with_ema)
result = run_backtest(data_with_position, initial_capital)

metrics = compute_metrics(result, initial_capital)

print("Final capital:", metrics['final_capital'])
print("Total return (%):", metrics['total_return'] * 100)
print("Max drawdown (%):", metrics['max_drawdown'] * 100)

plot_price_and_ema(result, ticker)
plot_equity_curve(result)
