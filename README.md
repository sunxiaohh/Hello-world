# FuturesDeskAPI Python Wrapper

A Python package for interacting with the TheFuturesDesk API, including authentication, account management, order placement, position management, trade history, contract search, historical bars, and real-time streaming via SignalR.

---

## Installation

Install from your local directory (editable mode recommended for development):

```bash
pip install -e .
```

Or standard install:

```bash
pip install .
```

**Dependencies:**

- `requests`
- `signalrcore` (for real-time streaming)

---

## Usage

### 1. Import and Authenticate

```python
from futuresdeskapi import FuturesDeskClient

username = "YOUR_USERNAME"
api_key = "YOUR_API_KEY"

client = FuturesDeskClient(username, api_key)
```

---

### 2. Accounts

```python
# List active accounts
accounts = client.account.search_accounts()
print(accounts)
```

---

### 3. Contracts

```python
# Search for contracts
contracts = client.contract.search_contracts("NQ")
print(contracts)

# Search for contract by ID
contract = client.contract.search_contract_by_id("CON.F.US.ENQ.H25")
print(contract)
```

---

### 4. Orders

```python
# Search for orders
orders = client.order.search_orders(account_id=123, start_timestamp="2025-01-01T00:00:00Z")

# Search for open orders
open_orders = client.order.search_open_orders(account_id=123)

# Place an order
order_id = client.order.place_order(
    account_id=123,
    contract_id="CON.F.US.DA6.M25",
    type=2,  # Market
    side=1,  # Sell
    size=1
)

# Cancel an order
client.order.cancel_order(account_id=123, order_id=456)

# Modify an order
client.order.modify_order(account_id=123, order_id=456, size=2, stop_price=1604)
```

---

### 5. Positions

```python
# Search for open positions
positions = client.position.search_open_positions(account_id=123)

# Close a position
client.position.close_position(account_id=123, contract_id="CON.F.US.GMET.J25")

# Partially close a position
client.position.partial_close_position(account_id=123, contract_id="CON.F.US.GMET.J25", size=1)
```

---

### 6. Trades

```python
# Search for trades
trades = client.trade.search_trades(account_id=123, start_timestamp="2025-01-01T00:00:00Z")
```

---

### 7. Historical Bars

```python
bars = client.history.retrieve_bars(
    contract_id="CON.F.US.RTY.Z24",
    live=False,
    start_time="2024-12-01T00:00:00Z",
    end_time="2024-12-31T21:00:00Z",
    unit=3,  # Hour
    unit_number=1,
    limit=7,
    include_partial_bar=False
)
```

---

### 8. Real-Time Streaming (SignalR)

> **Note:** Real-time streaming requires `signalrcore` and is optional.

```python
# Start real-time connection
client.realtime.start()

# Subscribe to account/order/position/trade updates
client.realtime.subscribe_accounts()
client.realtime.subscribe_orders(account_id=123)
client.realtime.subscribe_positions(account_id=123)
client.realtime.subscribe_trades(account_id=123)

# Register event handlers
def on_order_update(data):
    print("Order update:", data)

client.realtime.on_order_update(on_order_update)
```

---

### 9. OCO (One-Cancels-the-Other) Orders

This client provides a **client-side OCO (One-Cancels-the-Other) implementation**. The underlying trading API may not support OCO orders natively. This means the `FuturesDeskClient` places two separate orders (typically a stop-loss and a take-profit) and then monitors their status and the status of the associated position. If one order executes, or if the overall position for the contract is closed, the client will automatically attempt to cancel the other remaining order of the OCO pair.

#### Managing the OCO Polling Service

The client-side OCO functionality relies on a polling service that periodically checks order and position statuses. You need to start this service for OCO orders to be managed:

```python
# Start the OCO manager (typically after initializing the client)
# You can optionally specify a polling interval in seconds (default is 10 seconds)
client.start_oco_manager(polling_interval=5) 

# ... your trading logic ...

# Stop the OCO manager before your application exits
# This is important to ensure graceful shutdown of the polling thread.
client.stop_oco_manager()
```

#### Placing an OCO Order

Here's how to place an OCO order using the client:

```python
# Example: Placing OCO orders for an existing/intended LONG position on 'CONTRACT_X'
# Assumes you want to protect a position of size 1.
# If current price is 100, set SL at 95 and TP at 105.
try:
    sl_order_id, tp_order_id, client_oco_id = client.place_oco_order(
        account_id=YOUR_ACCOUNT_ID, # Replace with your actual account ID
        contract_id="CONTRACT_X",   # Replace with the actual contract ID
        position_side=1,  # 1 for Buy/Long position, 2 for Sell/Short position
        size=1,
        stop_loss_price=95.0,
        take_profit_price=105.0,
        custom_tag_sl="my_sl_order_123", # Optional custom tag for the stop-loss order
        custom_tag_tp="my_tp_order_123"  # Optional custom tag for the take-profit order
    )
    print(f"OCO order placed: SL ID {sl_order_id}, TP ID {tp_order_id}, Managed ID {client_oco_id}")
except Exception as e:
    print(f"Error placing OCO order: {e}")
    # Remember that if the SL leg was placed but TP failed, the exception from
    # place_oco_order in OrderAPI (and subsequently FuturesDeskClient)
    # will contain the SL order ID for manual cancellation if needed.
```

#### Important Caveats

-   **Client-Side Logic:** The OCO management logic runs entirely within your client application. If your application is not running, crashes, or loses internet connectivity, the OCO orders will **not** be managed.
-   **Polling Delay:** Management of OCO orders depends on the `polling_interval`. There will be a delay (up to the polling interval) between an event (e.g., one leg of the OCO filling, or the position being closed manually) and the client attempting to cancel the corresponding OCO leg.
-   **API Errors & Rate Limits:** The OCO manager makes API calls to check order/position statuses and to cancel orders. Failures in these API calls (due to network issues, API errors, or rate limits) can affect or delay OCO management.
-   **No Guarantees:** Due to its client-side nature, polling delays, and potential for API communication failures, this implementation **cannot offer the same guarantees** as native server-side OCO orders. Use with a clear understanding of these limitations.

---

## Notes

- All API requests require a valid session token, handled automatically after login.
- Session tokens are valid for 24 hours; re-authenticate as needed.
- For more details on endpoints and parameters, refer to the official TheFuturesDesk API documentation.

---

## License

MIT License
