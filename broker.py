balance = 10000
open_trade = None

def place_order(signal, price, size):
    global open_trade

    if open_trade is not None:
        return "Trade already open"

    if "BUY" in signal:
        open_trade = {
            "type": "BUY",
            "entry": price,
            "size": size
        }
        return f"BUY ORDER FILLED @ {price}"

    if "SELL" in signal:
        open_trade = {
            "type": "SELL",
            "entry": price,
            "size": size
        }
        return f"SELL ORDER FILLED @ {price}"

    return "NO ORDER"

def check_trade(price):
    global balance, open_trade

    if open_trade is None:
        return

    pnl = 0
    if open_trade["type"] == "BUY":
        pnl = (price - open_trade["entry"]) * open_trade["size"]
    else:
        pnl = (open_trade["entry"] - price) * open_trade["size"]

    balance += pnl
    open_trade = None
    return f"Trade closed | PnL: {round(pnl,2)} | Balance: {round(balance,2)}"