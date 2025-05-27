import os
import time
import json
import logging
import threading
import signal
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from futuresdeskapi import FuturesDeskClient
from futuresdeskapi.realtime import RealTimeClient # For type hinting if needed

# --- Constants ---
CONFIG_FILE = "config.json"
STATE_FILE = "trading_state.json"
LOG_FILE = "trading_bot.log"
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5
HEARTBEAT_INTERVAL_SECONDS = 60 # For main loop
POSITION_CHECK_INTERVAL_SECONDS = 15 # How often to check current position before placing new trades
TRADE_COOLDOWN_SECONDS = 5 # Minimum time between trades for the same contract

# --- Configuration & Globals ---
config = {}
trading_state = {} # Stores active trade IDs, positions, etc.
futures_client = None # type: FuturesDeskClient | None
realtime_client = None # type: RealTimeClient | None
stop_event = threading.Event()
last_trade_timestamps = {} # contract_id -> timestamp

# --- Logging Setup ---
# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def load_config():
    global config
    if not os.path.exists(CONFIG_FILE):
        logger.error(f"CRITICAL: {CONFIG_FILE} not found. Please create it.")
        exit(1)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    logger.info("Configuration loaded successfully.")

def save_state():
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(trading_state, f, indent=4)
        logger.debug("Trading state saved.")
    except Exception as e:
        logger.error(f"Error saving trading state: {e}")

def load_state():
    global trading_state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                trading_state = json.load(f)
            logger.info("Trading state loaded.")
        except Exception as e:
            logger.error(f"Error loading trading state: {e}. Starting with a fresh state.")
            trading_state = {}
    else:
        logger.info("No existing state file found. Starting with a fresh state.")
        trading_state = {}

def get_env_variable(var_name, default=None):
    val = os.getenv(var_name)
    if val is None:
        logger.warning(f"Environment variable {var_name} not found.")
        if default is not None:
            logger.warning(f"Using default value for {var_name}: {default}")
            return default
        else:
            logger.error(f"CRITICAL: Environment variable {var_name} is required but not set. Exiting.")
            exit(1)
    return val

# --- API Interaction ---
def initialize_futuresdesk_api():
    global futures_client
    username = get_env_variable("FUTURES_DESK_USERNAME")
    api_key = get_env_variable("FUTURES_DESK_API_KEY")
    base_url = config.get("api_base_url", "https://api.thefuturesdesk.projectx.com") # Default if not in config

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempting to connect to FuturesDesk API (Attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})...")
            futures_client = FuturesDeskClient(username=username, api_key=api_key, base_url=base_url)
            # Test connection by fetching accounts
            accounts = futures_client.account.search_accounts()
            if accounts:
                logger.info(f"Successfully connected to FuturesDesk API. Account ID: {accounts[0]['id']}") # Assumes at least one account
                trading_state['account_id'] = accounts[0]['id'] # Store the primary account ID
                return True
            else:
                logger.error("Connected to API but no accounts found.")
                return False # Or handle as appropriate
        except Exception as e:
            logger.error(f"Error initializing FuturesDesk API: {e}")
            if attempt < MAX_RETRY_ATTEMPTS - 1:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Max retry attempts reached for API initialization.")
                return False
    return False

def get_current_position(contract_id):
    if not futures_client or 'account_id' not in trading_state:
        logger.error("API client or account ID not initialized for get_current_position.")
        return None
    try:
        # Ensure account_id is passed as integer if API expects it
        positions = futures_client.position.search_open_positions(account_id=int(trading_state['account_id']))
        for pos in positions:
            if pos['contractId'] == contract_id:
                logger.debug(f"Current position for {contract_id}: Size {pos.get('size', 0)}, Avg Price {pos.get('avgPrice', 0)}")
                return pos # Returns the position dictionary
        logger.debug(f"No open position found for {contract_id}.")
        return None # No position for this specific contract
    except Exception as e:
        logger.error(f"Error fetching position for {contract_id}: {e}")
        return None

def place_market_order(contract_id, side, size, strategy_tag=""):
    if not futures_client or 'account_id' not in trading_state:
        logger.error("API client or account ID not initialized for place_market_order.")
        return None

    # Cooldown check
    now = time.time()
    if contract_id in last_trade_timestamps and \
       now - last_trade_timestamps[contract_id] < TRADE_COOLDOWN_SECONDS:
        logger.warning(f"Trade cooldown for {contract_id}. Skipping order.")
        return None

    custom_tag = f"Bot_{strategy_tag}_{int(time.time())}" if strategy_tag else f"Bot_{int(time.time())}"

    logger.info(f"Placing MARKET order: {contract_id}, Side: {'BUY' if side == 1 else 'SELL'}, Size: {size}, Tag: {custom_tag}")
    try:
        order_id = futures_client.order.place_order(
            account_id=int(trading_state['account_id']), # Ensure account_id is int
            contract_id=contract_id,
            type=2,  # Market Order
            side=side, # 1 for Buy, 2 for Sell
            size=int(size), # Ensure size is int
            custom_tag=custom_tag
        )
        logger.info(f"Market order placed successfully. Order ID: {order_id} for {contract_id}")
        last_trade_timestamps[contract_id] = now
        trading_state.setdefault('active_orders', []).append({
            'order_id': order_id,
            'contract_id': contract_id,
            'side': side,
            'size': size,
            'status': 'open', # Initial status
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        save_state()
        return order_id
    except Exception as e:
        logger.error(f"Error placing market order for {contract_id}: {e}")
        return None

# This is the function that uses the client library's OCO placement
def place_oco_orders_for_position(contract_id, position_side, position_size, sl_price, tp_price, strategy_tag=""):
    if not futures_client or 'account_id' not in trading_state:
        logger.error("API client or account ID not initialized for place_oco_orders_for_position.")
        return None, None, None # Return three values to match expected unpacking

    custom_tag_base = f"BotOCO_{strategy_tag}_{int(time.time())}" if strategy_tag else f"BotOCO_{int(time.time())}"
    sl_tag = f"{custom_tag_base}_SL"
    tp_tag = f"{custom_tag_base}_TP"

    logger.info(f"Placing OCO orders for {contract_id}: PosSide {'LONG' if position_side == 1 else 'SHORT'}, Size {position_size}, SL {sl_price}, TP {tp_price}")
    try:
        sl_order_id, tp_order_id, client_oco_id = futures_client.place_oco_order(
            account_id=int(trading_state['account_id']), 
            contract_id=contract_id,
            position_side=position_side, # 1 for Long, 2 for Short
            size=int(position_size), 
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            custom_tag_sl=sl_tag,
            custom_tag_tp=tp_tag
        )
        logger.info(f"OCO orders placed for {contract_id}: SL ID {sl_order_id}, TP ID {tp_order_id}. Managed by Client OCO ID: {client_oco_id}")
        # Store OCO group info, including the main entry order ID if available (though not passed to this func)
        trading_state.setdefault('active_oco_groups', {})[client_oco_id] = {
            'sl_order_id': sl_order_id,
            'tp_order_id': tp_order_id,
            'contract_id': contract_id,
            'account_id': trading_state['account_id'], 
            'status': 'active', # This status is managed by the library's OCO manager
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        save_state()
        return sl_order_id, tp_order_id, client_oco_id
    except Exception as e:
        logger.error(f"Error placing OCO orders for {contract_id}: {e}")
        # Check if SL was placed before TP failed (from library's exception message)
        if "Stop-loss order (ID:" in str(e): 
             try:
                sl_order_id_partial = int(str(e).split("Stop-loss order (ID: ")[1].split(")")[0])
                logger.warning(f"OCO placement failed for TP, but SL order {sl_order_id_partial} was placed. Attempting to cancel SL order {sl_order_id_partial}.")
                futures_client.order.cancel_order(account_id=int(trading_state['account_id']), order_id=sl_order_id_partial)
                logger.info(f"Successfully cancelled orphaned SL order {sl_order_id_partial}.")
             except Exception as cancel_e:
                logger.error(f"Failed to cancel orphaned SL order {sl_order_id_partial} after OCO TP failure: {cancel_e}")
        return None, None, None


# --- Real-Time Event Handlers ---
def on_account_update(data):
    logger.info(f"Realtime Account Update: {data}")

def on_order_update(data):
    logger.info(f"Realtime Order Update: {data}")
    order_id = str(data.get('id'))
    status = data.get('status') 

    with threading.Lock(): 
        updated = False
        for order in trading_state.get('active_orders', []):
            if str(order.get('order_id')) == order_id:
                order['status'] = status
                order['update_timestamp'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Order {order_id} status updated to {status} in active_orders.")
                updated = True
                break
        
        # The client library's OCO manager handles OCO leg updates internally.
        # This log is just for awareness if an OCO leg update comes through general order stream.
        for oco_id, oco_group in trading_state.get('active_oco_groups', {}).items():
            if str(oco_group.get('sl_order_id')) == order_id or str(oco_group.get('tp_order_id')) == order_id:
                leg_type = "SL" if str(oco_group.get('sl_order_id')) == order_id else "TP"
                logger.info(f"Order update for OCO leg: OCO ID {oco_id}, Leg {leg_type} ({order_id}), New Status: {status}")
                updated = True 
                break
        if updated:
            save_state()

def on_position_update(data):
    logger.info(f"Realtime Position Update: {data}")
    contract_id = data.get('contractId')
    current_size = data.get('size', 0) # Assuming size is float or can be parsed to float
    with threading.Lock():
        trading_state.setdefault('current_positions', {})[contract_id] = data
        save_state()
    logger.info(f"Position for {contract_id} updated in state: Size {current_size}")
    # The client library's OCO manager should react to position closures.
    if float(current_size) == 0.0:
        logger.info(f"Position for {contract_id} is now closed (Size: 0). OCO manager should handle related OCOs.")

def on_trade_update(data):
    logger.info(f"Realtime Trade Update (Execution): {data}")
    order_id = str(data.get('orderId'))
    contract_id = data.get('contractId')
    with threading.Lock():
        trading_state.setdefault('trade_executions', []).append(data)
        for order in trading_state.get('active_orders', []):
            if str(order.get('order_id')) == order_id:
                if order['status'] != "Filled": 
                    order['status'] = "Filled" 
                    order['fill_timestamp'] = datetime.now(timezone.utc).isoformat()
                    logger.info(f"Order {order_id} for {contract_id} marked as Filled due to trade execution.")
                break
        save_state()

def initialize_realtime_client():
    if not futures_client or not futures_client.token:
        logger.error("FuturesDeskClient not initialized or not authenticated. Cannot start RealTimeClient.")
        return False
    try:
        logger.info("Initializing RealTimeClient...")
        if not hasattr(futures_client, 'realtime') or futures_client.realtime is None:
            logger.error("Realtime client component not found in FuturesDeskClient instance.")
            return False

        futures_client.realtime.on_account_update(on_account_update)
        futures_client.realtime.on_order_update(on_order_update)
        futures_client.realtime.on_position_update(on_position_update)
        futures_client.realtime.on_trade_update(on_trade_update)
        
        logger.info("Starting RealTimeClient connection...")
        futures_client.realtime.start() 
        time.sleep(config.get("realtime_subscribe_delay_seconds", 5)) 

        if 'account_id' in trading_state:
            logger.info(f"Subscribing to realtime updates for account ID: {trading_state['account_id']}")
            futures_client.realtime.subscribe_accounts() 
            futures_client.realtime.subscribe_orders(int(trading_state['account_id']))
            futures_client.realtime.subscribe_positions(int(trading_state['account_id']))
            futures_client.realtime.subscribe_trades(int(trading_state['account_id']))
            logger.info("Realtime subscriptions sent.")
        else:
            logger.error("Account ID not available in trading_state. Cannot subscribe to realtime events.")
            return False
            
        logger.info("RealTimeClient initialized and subscriptions sent.")
        return True
    except Exception as e:
        logger.error(f"Error initializing RealTimeClient: {e}")
        return False

# --- Trading Strategy ---
def simple_momentum_strategy():
    """
    Refactored momentum strategy with position reversal and OCO placement
    using futures_client.place_oco_order via place_oco_orders_for_position helper.
    """
    logger.info("Evaluating simple_momentum_strategy...")
    enabled_strategies = config.get("enabled_strategies", [])
    if "simple_momentum" not in enabled_strategies:
        logger.info("Simple momentum strategy is not enabled in config.json.")
        return

    strategy_config = config.get("strategy_simple_momentum", {})
    contracts_to_trade = strategy_config.get("contracts", [])
    
    if not contracts_to_trade:
        logger.warning("No contracts specified for simple_momentum strategy in config.json.")
        return

    for contract_details in contracts_to_trade:
        target_contract_id = contract_details.get("id") 
        if not target_contract_id:
            logger.warning("Contract details missing 'id' in strategy_simple_momentum config. Skipping.")
            continue
        
        # --- 1. Fetch Current Position ---
        existing_position = get_current_position(target_contract_id)
        current_position_direction = "FLAT"
        existing_position_size = 0.0 

        # --- 2. Determine Existing Position Side (using size) ---
        if existing_position and existing_position.get('size') is not None:
            size_from_api = existing_position.get('size', 0.0)
            try: 
                existing_position_size = float(size_from_api)
            except ValueError:
                logger.error(f"{target_contract_id}: Could not parse position size '{size_from_api}' to float.")
                existing_position_size = 0.0

            if existing_position_size > 0:
                current_position_direction = "LONG"
            elif existing_position_size < 0:
                current_position_direction = "SHORT"
        
        logger.info(f"{target_contract_id}: Current position: {current_position_direction}, Size: {existing_position_size}")

        # --- 3. Simulated Gemini Signal ---
        import random
        new_signal_action = random.choice(["BUY", "SELL", "HOLD"]) 
        current_market_price_placeholder = 100.00 # Placeholder for SL/TP calculation
        logger.info(f"{target_contract_id}: New signal: {new_signal_action}. Placeholder current price for SL/TP: {current_market_price_placeholder}")
        
        # --- Strategy Parameters ---
        position_size_to_trade = contract_details.get("position_size", 1) 
        sl_offset_val = contract_details.get("stop_loss_offset_ticks", 10) * contract_details.get("tick_size", 0.25)
        tp_offset_val = contract_details.get("take_profit_offset_ticks", 20) * contract_details.get("tick_size", 0.25)

        # --- Cooldown Check ---
        now = time.time()
        if target_contract_id in last_trade_timestamps and \
           now - last_trade_timestamps[target_contract_id] < TRADE_COOLDOWN_SECONDS:
            logger.info(f"{target_contract_id}: Trade cooldown active. Skipping evaluation cycle.")
            continue

        proceed_to_new_trade_action = None 
        entry_after_reversal = False 

        # --- 4. Implement Reversal Logic ---
        if current_position_direction == "LONG" and new_signal_action == "SELL":
            logger.info(f"Reversal signal for {target_contract_id}: Closing existing LONG position (Size: {existing_position_size}) to go SHORT.")
            try:
                # Ensure account_id is int for API call
                close_success = futures_client.position.close_position(account_id=int(trading_state['account_id']), contract_id=target_contract_id)
                if close_success: 
                    logger.info(f"Request to close LONG position for {target_contract_id} sent successfully.")
                    time.sleep(config.get("reversal_flatten_delay_seconds", 2)) # Wait for position to potentially flatten
                    proceed_to_new_trade_action = "SELL"
                    entry_after_reversal = True 
                else:
                    logger.error(f"Failed to send request to close LONG position for {target_contract_id}. Not reversing.")
            except Exception as e:
                logger.error(f"Error closing LONG position for {target_contract_id}: {e}. Not reversing.")
        
        elif current_position_direction == "SHORT" and new_signal_action == "BUY":
            logger.info(f"Reversal signal for {target_contract_id}: Closing existing SHORT position (Size: {existing_position_size}) to go LONG.")
            try:
                close_success = futures_client.position.close_position(account_id=int(trading_state['account_id']), contract_id=target_contract_id)
                if close_success:
                    logger.info(f"Request to close SHORT position for {target_contract_id} sent successfully.")
                    time.sleep(config.get("reversal_flatten_delay_seconds", 2)) # Wait
                    proceed_to_new_trade_action = "BUY"
                    entry_after_reversal = True
                else:
                    logger.error(f"Failed to send request to close SHORT position for {target_contract_id}. Not reversing.")
            except Exception as e:
                logger.error(f"Error closing SHORT position for {target_contract_id}: {e}. Not reversing.")

        elif current_position_direction != "FLAT" and \
             ((new_signal_action == "BUY" and current_position_direction == "LONG") or \
              (new_signal_action == "SELL" and current_position_direction == "SHORT")):
            logger.info(f"{target_contract_id}: New signal {new_signal_action} matches current {current_position_direction} position. No new entry action taken.")
            last_trade_timestamps[target_contract_id] = now 
            continue # Skip to next contract in the loop

        elif new_signal_action == "HOLD":
            logger.info(f"{target_contract_id}: Signal is HOLD. No action taken.")
            last_trade_timestamps[target_contract_id] = now
            continue # Skip to next contract
        
        elif current_position_direction == "FLAT" and (new_signal_action == "BUY" or new_signal_action == "SELL"):
            proceed_to_new_trade_action = new_signal_action # New entry from flat position

        # --- 5. Integrate OCO Placement after Entry Order ---
        if proceed_to_new_trade_action: # This is "BUY" or "SELL"
            entry_side_numeric = 1 if proceed_to_new_trade_action == "BUY" else 2
            oco_position_side_numeric = 1 if proceed_to_new_trade_action == "BUY" else 2 # 1 for Long, 2 for Short for OCO
            
            log_message_prefix = f"{target_contract_id}: "
            if entry_after_reversal:
                log_message_prefix += f"Continuing with {proceed_to_new_trade_action} leg of reversal. "
            else: # New entry from flat
                log_message_prefix += f"Proceeding to place new {proceed_to_new_trade_action} entry order from FLAT. "
            logger.info(f"{log_message_prefix}Size: {position_size_to_trade}.")

            entry_order_id = place_market_order(
                contract_id=target_contract_id, 
                side=entry_side_numeric, 
                size=int(position_size_to_trade), # Ensure size is int
                strategy_tag="momentum_entry"
            )

            if entry_order_id:
                logger.info(f"{target_contract_id}: Entry order {entry_order_id} placed successfully.")
                last_trade_timestamps[target_contract_id] = now # Update timestamp after successful entry

                # OCO Placement
                entry_fill_price_placeholder = current_market_price_placeholder # Using the placeholder price
                logger.warning(f"{target_contract_id}: Using placeholder entry price ({entry_fill_price_placeholder}) for SL/TP calculation. "
                               "This is NOT suitable for live trading. Implement actual fill price retrieval from trade execution for accuracy.")
                
                if proceed_to_new_trade_action == "BUY": # New position is LONG
                    sl_price = entry_fill_price_placeholder - sl_offset_val
                    tp_price = entry_fill_price_placeholder + tp_offset_val
                else: # New position is SHORT (proceed_to_new_trade_action == "SELL")
                    sl_price = entry_fill_price_placeholder + sl_offset_val
                    tp_price = entry_fill_price_placeholder - tp_offset_val
                
                logger.info(f"{target_contract_id}: Calculated SL: {sl_price}, TP: {tp_price} for new {proceed_to_new_trade_action} position based on placeholder price.")
                
                sl_oco_id, tp_oco_id, client_oco_id = place_oco_orders_for_position(
                    contract_id=target_contract_id,
                    position_side=oco_position_side_numeric, # Correct side for OCO
                    position_size=int(position_size_to_trade), # Ensure size is int
                    sl_price=sl_price,
                    tp_price=tp_price,
                    strategy_tag="momentum_oco" # Pass strategy tag for OCO orders
                )

                if sl_oco_id and tp_oco_id:
                    logger.info(f"{target_contract_id}: OCO orders placed successfully via helper. SL ID: {sl_oco_id}, TP ID: {tp_oco_id}, Client OCO ID: {client_oco_id}")
                    # The place_oco_orders_for_position function already handles saving state for the OCO group
                else:
                    logger.error(f"{target_contract_id}: Failed to place OCO orders using helper function after entry order {entry_order_id}.")
                    logger.warning(f"{target_contract_id}: Main entry order {entry_order_id} is active without OCO protection. "
                                   "Consider manual intervention or automated cancellation of the main order.")
            else: # Entry order placement failed
                logger.error(f"{target_contract_id}: Failed to place {proceed_to_new_trade_action} entry order. OCO orders will not be placed.")
                last_trade_timestamps[target_contract_id] = now # Update timestamp to respect cooldown
        else: # No new trade action determined (e.g. HOLD signal, or reversal failed to flatten)
            if new_signal_action != "HOLD": # Avoid redundant log if it was already HOLD
                 logger.info(f"{target_contract_id}: No new trade action taken. Signal: {new_signal_action}, Current Pos: {current_position_direction}, Proceed Action: {proceed_to_new_trade_action}")
            last_trade_timestamps[target_contract_id] = now # Update timestamp to respect cooldown

    logger.info("simple_momentum_strategy evaluation complete.")


# --- Main Application Logic & Lifecycle ---
def run_trading_bot():
    logger.info("Starting trading bot...") 
    
    if config.get("enable_oco_manager", True): 
        if hasattr(futures_client, 'start_oco_manager'):
            oco_polling_interval = config.get("oco_manager_polling_interval_seconds", 10) 
            logger.info(f"Starting Client Library's OCO Manager with polling interval: {oco_polling_interval}s")
            futures_client.start_oco_manager(polling_interval=oco_polling_interval)
        else:
            logger.warning("Client Library's OCO Manager (start_oco_manager) not found on futures_client. OCOs via client lib will not be managed.")

    if config.get("enable_realtime_client", True): 
        if not initialize_realtime_client(): 
            logger.warning("Failed to initialize RealTimeClient. Some features might be limited (e.g. live updates).")

    last_heartbeat_time = time.time()
    while not stop_event.is_set():
        try:
            current_time = time.time()
            if current_time - last_heartbeat_time > HEARTBEAT_INTERVAL_SECONDS:
                logger.info(f"Bot is alive. Account ID: {trading_state.get('account_id')}. Active OCOs (from state): {len(trading_state.get('active_oco_groups', {}))}")
                last_heartbeat_time = current_time
            
            simple_momentum_strategy() 

            save_state() 
            stop_event.wait(config.get("main_loop_interval_seconds", 60)) 
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in main loop. Initiating shutdown...")
            stop_event.set()
        except Exception as e:
            logger.error(f"Unhandled exception in main trading loop: {e}", exc_info=True)
            time.sleep(RETRY_DELAY_SECONDS) 
    logger.info("Trading bot main loop terminated.")


def signal_handler(signum, frame):
    logger.info(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown...")
    stop_event.set()
    if futures_client and hasattr(futures_client, 'stop_oco_manager'):
        try:
            logger.info("Signal Handler: Attempting to stop Client Library's OCO Manager...")
            futures_client.stop_oco_manager()
            logger.info("Signal Handler: Client Library's OCO Manager stop signal sent.")
        except Exception as e:
            logger.error(f"Signal Handler: Error stopping Client Library's OCO Manager: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    load_dotenv() 
    load_config()
    load_state() 

    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler) 

    if not initialize_futuresdesk_api():
        logger.critical("CRITICAL: Failed to initialize FuturesDesk API. Exiting.")
        exit(1)
    
    run_trading_bot()

    # --- Shutdown Sequence ---
    logger.info("Shutting down bot...")
    
    if config.get("enable_oco_manager", True) and hasattr(futures_client, 'stop_oco_manager'):
        try:
            logger.info("Stopping Client Library's OCO Manager...")
            futures_client.stop_oco_manager()
            logger.info("Client Library's OCO Manager stopped.")
        except Exception as e:
            logger.error(f"Error stopping Client Library's OCO Manager: {e}")

    if config.get("enable_realtime_client", True) and hasattr(futures_client, 'realtime') and futures_client.realtime and hasattr(futures_client.realtime, 'stop'):
        try:
            logger.info("Stopping RealTimeClient...")
            futures_client.realtime.stop() 
            logger.info("RealTimeClient stopped.")
        except Exception as e:
            logger.error(f"Error stopping RealTimeClient: {e}")
    
    save_state() 
    logger.info("FuturesDesk Trading Bot has terminated gracefully.")
