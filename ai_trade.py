import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
import sched
import signal
import pytz
import mplfinance as mpf
import threading
from typing import Dict, List, Optional, Any

# --- Configuration ---
# FuturesDesk credentials
FUTURESDESK_USERNAME = "xiaosun666"
FUTURESDESK_API_KEY = "MmP9cmQB8Zra4UVLQnQUipyIeBDObzyLmu/NZdkwHUA="

os.environ["GOOGLE_API_KEY"] = "AIzaSyD5lHkApBOPhiuLLdMf02M8c74ggJDy0KU"

CONTRACT_SYMBOL = "MNQ"
PLOTS_DIR = "trading_charts"
OCO_RECORDS_DIR = "oco_records"
NUMBER_OF_BARS_TO_PLOT = 100
DAYS_OF_DATA_TO_FETCH = 60
BAR_LIMIT_PER_FETCH = 500
GEMINI_VISION_MODEL = "gemini-2.0-flash"

# Scheduling Configuration
TARGET_START_HOUR_EST = 9
TARGET_START_MINUTE_EST = 45
CYCLE_INTERVAL_MINUTES = 5

# Risk Management Configuration
MAX_RISK_PER_TRADE = 150
MAX_CONTRACTS = 6
MAX_TOTAL_CONTRACTS = 15  # Maximum total contracts across all positions
OCO_CHECK_INTERVAL_SECONDS = 5  # How often to check OCO orders

# --- Global variables ---
futuresdesk_client = None
gemini_model = None
ACCOUNT_ID = None
CONTRACT_ID = None
CONTRACT_DISPLAY_SYMBOL = CONTRACT_SYMBOL

s = sched.scheduler(time.time, time.sleep)
shutdown_flag = False
first_run_after_start_time_done = False
api_call_times = []

# OCO Management
oco_manager = None
oco_monitor_thread = None
oco_lock = threading.Lock()

# --- Create directories ---
for directory in [PLOTS_DIR, OCO_RECORDS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("futuresdesk_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- OCO Order Management System ---
class OCOOrderManager:
    """
    Manages One-Cancels-Other (OCO) orders for futures trading.
    Tracks active trades and automatically cancels remaining orders when one is filled.
    """
    
    def __init__(self, client, account_id: int):
        self.client = client
        self.account_id = account_id
        self.active_oco_groups: Dict[str, Dict] = {}
        self.trade_records: List[Dict] = []
        self.is_monitoring = False
        
    def create_oco_group(self, position_id: str, main_order_id: int, 
                        stop_loss_order_id: Optional[int] = None, 
                        take_profit_order_id: Optional[int] = None,
                        contract_id: str = None, trade_side: str = None,
                        entry_price: float = None) -> str:
        """
        Create a new OCO group for tracking orders.
        
        Args:
            position_id: Unique identifier for this trade position
            main_order_id: ID of the main entry order
            stop_loss_order_id: ID of the stop loss order (optional)
            take_profit_order_id: ID of the take profit order (optional)
            contract_id: Contract being traded
            trade_side: 'LONG' or 'SHORT'
            entry_price: Entry price for the trade
            
        Returns:
            str: The position_id used as the group key
        """
        with oco_lock:
            oco_group = {
                'position_id': position_id,
                'position_active': True,
                'main_order_id': main_order_id,
                'stop_loss_order_id': stop_loss_order_id,
                'take_profit_order_id': take_profit_order_id,
                'contract_id': contract_id,
                'trade_side': trade_side,
                'entry_price': entry_price,
                'created_timestamp': datetime.now().isoformat(),
                'main_order_filled': False,
                'exit_order_filled': None,  # Will be 'SL' or 'TP' when filled
                'all_orders_status': {},
                'position_size': 0,
                'unrealized_pnl': 0.0,
                'notes': []
            }
            
            self.active_oco_groups[position_id] = oco_group
            logger.info(f"Created OCO group {position_id}: Main={main_order_id}, SL={stop_loss_order_id}, TP={take_profit_order_id}")
            
            # Save initial record
            self._save_oco_record(oco_group.copy())
            
        return position_id
    
    def update_order_status(self, position_id: str, order_id: int, status: str, filled_price: float = None):
        """
        Update the status of an order in an OCO group.
        
        Args:
            position_id: The OCO group identifier
            order_id: The order ID that was updated
            status: New status ('FILLED', 'CANCELLED', 'PENDING', etc.)
            filled_price: Price at which order was filled (if applicable)
        """
        with oco_lock:
            if position_id not in self.active_oco_groups:
                logger.warning(f"OCO group {position_id} not found for order update")
                return
                
            oco_group = self.active_oco_groups[position_id]
            oco_group['all_orders_status'][order_id] = {
                'status': status,
                'updated_timestamp': datetime.now().isoformat(),
                'filled_price': filled_price
            }
            
            # Check if main order was filled
            if order_id == oco_group['main_order_id'] and status == 'FILLED':
                oco_group['main_order_filled'] = True
                oco_group['notes'].append(f"Main order {order_id} filled at {filled_price}")
                logger.info(f"Main order {order_id} filled for OCO group {position_id}")
                
            # Check if exit order was filled
            elif order_id == oco_group['stop_loss_order_id'] and status == 'FILLED':
                oco_group['exit_order_filled'] = 'SL'
                oco_group['position_active'] = False
                oco_group['notes'].append(f"Stop loss {order_id} filled at {filled_price}")
                logger.info(f"Stop loss order {order_id} filled for OCO group {position_id}")
                self._handle_position_closed(position_id, 'SL', filled_price)
                
            elif order_id == oco_group['take_profit_order_id'] and status == 'FILLED':
                oco_group['exit_order_filled'] = 'TP'
                oco_group['position_active'] = False
                oco_group['notes'].append(f"Take profit {order_id} filled at {filled_price}")
                logger.info(f"Take profit order {order_id} filled for OCO group {position_id}")
                self._handle_position_closed(position_id, 'TP', filled_price)
    
    def _handle_position_closed(self, position_id: str, exit_type: str, exit_price: float):
        """
        Handle cleanup when a position is closed by SL or TP.
        
        Args:
            position_id: The OCO group identifier
            exit_type: 'SL' or 'TP'
            exit_price: Price at which position was closed
        """
        if position_id not in self.active_oco_groups:
            return
            
        oco_group = self.active_oco_groups[position_id]
        
        # Cancel remaining orders
        orders_to_cancel = []
        if exit_type == 'SL' and oco_group['take_profit_order_id']:
            orders_to_cancel.append(oco_group['take_profit_order_id'])
        elif exit_type == 'TP' and oco_group['stop_loss_order_id']:
            orders_to_cancel.append(oco_group['stop_loss_order_id'])
            
        for order_id in orders_to_cancel:
            try:
                self.client.order.cancel_order(
                    account_id=self.account_id,
                    order_id=order_id
                )
                oco_group['all_orders_status'][order_id] = {
                    'status': 'CANCELLED_BY_OCO',
                    'updated_timestamp': datetime.now().isoformat(),
                    'filled_price': None
                }
                oco_group['notes'].append(f"Cancelled order {order_id} due to {exit_type} fill")
                logger.info(f"Cancelled order {order_id} for OCO group {position_id}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id} for OCO group {position_id}: {e}")
                oco_group['notes'].append(f"Failed to cancel order {order_id}: {str(e)}")
        
        # Calculate P&L if we have entry and exit prices
        if oco_group['entry_price'] and exit_price:
            pnl = self._calculate_pnl(oco_group['entry_price'], exit_price, 
                                    oco_group['trade_side'], oco_group['position_size'])
            oco_group['realized_pnl'] = pnl
            oco_group['notes'].append(f"Realized P&L: ${pnl:.2f}")
        
        # Final record save
        oco_group['closed_timestamp'] = datetime.now().isoformat()
        oco_group['exit_type'] = exit_type
        oco_group['exit_price'] = exit_price
        
        self._save_oco_record(oco_group.copy())
        
        # Move to trade records and remove from active
        self.trade_records.append(oco_group.copy())
        del self.active_oco_groups[position_id]
        
        logger.info(f"OCO group {position_id} completed and removed from active tracking")
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, trade_side: str, position_size: int) -> float:
        """Calculate P&L for a completed trade."""
        if trade_side.upper() == 'LONG':
            price_diff = exit_price - entry_price
        else:  # SHORT
            price_diff = entry_price - exit_price
            
        # Get contract specifications for P&L calculation
        contract_multiplier, tick_size, tick_value = get_instrument_details(CONTRACT_SYMBOL)
        return (price_diff / tick_size) * tick_value * position_size
    
    def monitor_positions(self):
        """
        Continuously monitor positions and update OCO groups.
        This should run in a separate thread.
        """
        self.is_monitoring = True
        logger.info("Starting OCO position monitoring")
        
        while self.is_monitoring and not shutdown_flag:
            try:
                with oco_lock:
                    active_groups = list(self.active_oco_groups.keys())
                
                for position_id in active_groups:
                    try:
                        self._check_oco_group_status(position_id)
                    except Exception as e:
                        logger.error(f"Error checking OCO group {position_id}: {e}")
                
                time.sleep(OCO_CHECK_INTERVAL_SECONDS)
                
            except Exception as e:
                logger.error(f"Error in OCO monitoring loop: {e}")
                time.sleep(OCO_CHECK_INTERVAL_SECONDS)
        
        logger.info("OCO position monitoring stopped")
    
    def _check_oco_group_status(self, position_id: str):
        """Check the status of orders and positions for an OCO group."""
        if position_id not in self.active_oco_groups:
            return
            
        oco_group = self.active_oco_groups[position_id]
        
        # Check open orders to see if any have been filled
        try:
            open_orders = self.client.order.search_open_orders(account_id=self.account_id)
            open_order_ids = {order.get('id') for order in open_orders if order.get('id')}
            
            # Check if any of our tracked orders are no longer open (might be filled)
            orders_to_check = [
                oco_group['main_order_id'],
                oco_group['stop_loss_order_id'],
                oco_group['take_profit_order_id']
            ]
            
            for order_id in orders_to_check:
                if order_id and order_id not in open_order_ids:
                    # Order is no longer open, check if it was filled
                    current_status = oco_group['all_orders_status'].get(order_id, {}).get('status')
                    if current_status != 'FILLED' and current_status != 'CANCELLED_BY_OCO':
                        # Assume it was filled (could also check order history)
                        self.update_order_status(position_id, order_id, 'FILLED')
            
            # Check current positions to update unrealized P&L
            positions = self.client.position.search_open_positions(account_id=self.account_id)
            for pos in positions:
                if pos.get('contract_id') == oco_group['contract_id']:
                    oco_group['position_size'] = pos.get('size', 0)
                    oco_group['unrealized_pnl'] = pos.get('unrealized_pnl', 0.0)
                    break
            else:
                # No position found, might have been closed
                if oco_group['main_order_filled'] and oco_group['position_active']:
                    logger.info(f"Position for OCO group {position_id} appears to have been closed")
                    # Mark as closed if we can't find the position
                    oco_group['position_active'] = False
                    if not oco_group['exit_order_filled']:
                        oco_group['exit_order_filled'] = 'MANUAL'
                        self._handle_position_closed(position_id, 'MANUAL', None)
                        
        except Exception as e:
            logger.error(f"Error checking status for OCO group {position_id}: {e}")
    
    def _save_oco_record(self, oco_record: Dict):
        """Save OCO record to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oco_record_{oco_record['position_id']}_{timestamp}.json"
            filepath = os.path.join(OCO_RECORDS_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(oco_record, f, indent=2, default=str)
                
            logger.debug(f"Saved OCO record to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save OCO record: {e}")
    
    def get_active_oco_groups(self) -> Dict:
        """Get copy of all active OCO groups."""
        with oco_lock:
            return self.active_oco_groups.copy()
    
    def force_close_oco_group(self, position_id: str, reason: str = "Manual close"):
        """Manually close an OCO group and cancel all orders."""
        with oco_lock:
            if position_id not in self.active_oco_groups:
                logger.warning(f"OCO group {position_id} not found for manual close")
                return
                
            oco_group = self.active_oco_groups[position_id]
            oco_group['position_active'] = False
            oco_group['notes'].append(f"Manually closed: {reason}")
            
            # Cancel all open orders
            orders_to_cancel = [
                oco_group['stop_loss_order_id'],
                oco_group['take_profit_order_id']
            ]
            
            for order_id in orders_to_cancel:
                if order_id:
                    try:
                        self.client.order.cancel_order(
                            account_id=self.account_id,
                            order_id=order_id
                        )
                        logger.info(f"Manually cancelled order {order_id}")
                    except Exception as e:
                        logger.error(f"Failed to manually cancel order {order_id}: {e}")
            
            self._handle_position_closed(position_id, 'MANUAL', None)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.is_monitoring = False


# --- API Initialization Functions ---
def initialize_futuresdesk_api():
    """Initialize FuturesDesk API client."""
    global futuresdesk_client, ACCOUNT_ID, CONTRACT_ID, CONTRACT_DISPLAY_SYMBOL
    logger.info("Initializing FuturesDesk API...")
    try:
        if not FUTURESDESK_USERNAME or FUTURESDESK_USERNAME == "YOUR_FUTURESDESK_USERNAME":
            logger.error("FuturesDesk Username not set.")
            return False
        if not FUTURESDESK_API_KEY or FUTURESDESK_API_KEY == "YOUR_FUTURESDESK_API_KEY":
            logger.error("FuturesDesk API Key not set.")
            return False

        from futuresdeskapi import FuturesDeskClient
        futuresdesk_client = FuturesDeskClient(username=FUTURESDESK_USERNAME, api_key=FUTURESDESK_API_KEY)
        logger.info("Successfully connected to FuturesDesk API.")

        accounts = futuresdesk_client.account.search_accounts()
        if accounts:
            ACCOUNT_ID = accounts[-1]['id']
            logger.info(f"Using Account ID: {ACCOUNT_ID}")
        else:
            logger.error("No active FuturesDesk accounts found.")
            return False

        contracts = futuresdesk_client.contract.search_contracts(search_text=CONTRACT_SYMBOL, live=False)
        if contracts:
            CONTRACT_ID = contracts[0]['id']
            CONTRACT_DISPLAY_SYMBOL = contracts[0].get('name', CONTRACT_SYMBOL)
            logger.info(f"Using Contract ID: {CONTRACT_ID} for {CONTRACT_DISPLAY_SYMBOL}")
            return True
        else:
            logger.error(f"Contract {CONTRACT_SYMBOL} not found on FuturesDesk.")
            return False
    except ImportError:
        logger.critical("futuresdeskapi library not found. Please install it: pip install futuresdeskapi")
        return False
    except Exception as e:
        logger.error(f"Error initializing FuturesDesk API: {e}", exc_info=True)
        return False

def initialize_gemini_api():
    """Initialize Gemini API."""
    global gemini_model
    logger.info("Initializing Gemini API...")
    try:
        gemini_api_key_env = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key_env:
            logger.warning("GOOGLE_API_KEY environment variable not set. Using hardcoded placeholder.")
            gemini_api_key = "YOUR_GEMINI_API_KEY_HERE"
            if gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
                 logger.critical("Gemini API Key is a placeholder. Update the script or set GOOGLE_API_KEY.")
                 return False
        else:
            gemini_api_key = gemini_api_key_env

        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        logger.info(f"Gemini API configured successfully using model: {GEMINI_VISION_MODEL}")
        return True
    except Exception as e:
        logger.error(f"Error initializing Gemini API: {e}", exc_info=True)
        return False

def initialize_oco_manager():
    """Initialize the OCO manager and start monitoring thread."""
    global oco_manager, oco_monitor_thread
    try:
        oco_manager = OCOOrderManager(futuresdesk_client, ACCOUNT_ID)
        
        # Start monitoring thread
        oco_monitor_thread = threading.Thread(target=oco_manager.monitor_positions, daemon=True)
        oco_monitor_thread.start()
        
        logger.info("OCO manager initialized and monitoring started")
        return True
    except Exception as e:
        logger.error(f"Error initializing OCO manager: {e}", exc_info=True)
        return False

# --- Utility Functions ---
def unit_name(unit_code):
    """Convert unit code to readable name."""
    if unit_code == 1: return "Minute"
    if unit_code == 3: return "Hour"
    return "UnknownUnit"

def get_instrument_details(contract_symbol):
    """Get contract-specific details needed for risk calculations."""
    contract_specs = {
        "MNQ": {"multiplier": 2, "tick_size": 0.25, "tick_value": 0.50},
        "MES": {"multiplier": 5, "tick_size": 0.25, "tick_value": 1.25},
        "ES": {"multiplier": 50, "tick_size": 0.25, "tick_value": 12.50},
        "NQ": {"multiplier": 20, "tick_size": 0.25, "tick_value": 5.00},
        "CL": {"multiplier": 1000, "tick_size": 0.01, "tick_value": 10.00},
        "GC": {"multiplier": 100, "tick_size": 0.10, "tick_value": 10.00},
        "ZB": {"multiplier": 1000, "tick_size": 1/32, "tick_value": 31.25},
    }
    
    default = {"multiplier": 1, "tick_size": 0.25, "tick_value": 0.25}
    contract_data = contract_specs.get(contract_symbol, default)
    return (
        contract_data["multiplier"],
        contract_data["tick_size"],
        contract_data["tick_value"]
    )

# --- Data Fetching and Analysis Functions ---
def fetch_historical_data(contract_id_to_fetch, unit, unit_number, days_back=DAYS_OF_DATA_TO_FETCH, limit_bars=BAR_LIMIT_PER_FETCH):
    """Fetch historical data from FuturesDesk API."""
    global futuresdesk_client
    if not futuresdesk_client:
        logger.error("FuturesDesk client not initialized for fetch_historical_data.")
        return pd.DataFrame()
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    logger.info(f"Fetching {unit_number} {unit_name(unit)} for {contract_id_to_fetch} from {start_time.isoformat()}Z to {end_time.isoformat()}Z")
    
    try:
        bars = futuresdesk_client.history.retrieve_bars(
            contract_id=contract_id_to_fetch, live=False,
            start_time=start_time.isoformat() + "Z", end_time=end_time.isoformat() + "Z",
            unit=unit, unit_number=unit_number, limit=limit_bars, include_partial_bar=False
        )
        if bars:
            df = pd.DataFrame(bars)
            column_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume','t':'timestamp'}
            df = df.rename(columns=column_map)
            if df.empty: return pd.DataFrame()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_index()
            logger.info(f"Fetched {len(df)} bars for {unit_number}{unit_name(unit)}.")
            return df
        else:
            logger.warning(f"No bars returned for {unit_number}{unit_name(unit)}.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data for {unit_number}{unit_name(unit)}: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate technical indicators."""
    if df.empty:
        logger.warning("DataFrame is empty. Cannot calculate indicators.")
        return df
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        logger.warning("DataFrame missing required OHLC columns for TA calculation.")
        if 'close' not in df.columns:
            return df

    df_out = df.copy()

    # 1. EMA (Exponential Moving Average) - 200 period for close price
    if 'close' in df_out.columns:
        try:
            df_out['EMA_200'] = df_out['close'].ewm(span=200, adjust=False, min_periods=200).mean()
            logger.info("Calculated EMA_200.")
        except Exception as e:
            logger.error(f"Error calculating EMA_200: {e}", exc_info=True)
            df_out['EMA_200'] = np.nan

    # 2. VWAP (Volume Weighted Average Price) - Daily Cumulative
    if 'high' in df_out.columns and 'low' in df_out.columns and 'close' in df_out.columns and 'volume' in df_out.columns:
        try:
            tp = (df_out['high'] + df_out['low'] + df_out['close']) / 3
            pv = tp * df_out['volume']

            if isinstance(df_out.index, pd.DatetimeIndex):
                date_col = df_out.index.date
                cum_pv_daily = pv.groupby(date_col).cumsum()
                cum_vol_daily = df_out['volume'].groupby(date_col).cumsum()
                df_out['VWAP'] = cum_pv_daily / cum_vol_daily
                df_out['VWAP'].ffill(inplace=True)
                logger.info("Calculated Daily VWAP.")
            else:
                logger.warning("DataFrame index is not a DatetimeIndex. Cannot calculate daily VWAP accurately. Calculating cumulative VWAP instead.")
                df_out['VWAP'] = (tp * df_out['volume']).cumsum() / df_out['volume'].cumsum()
                df_out['VWAP'].ffill(inplace=True)

        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}", exc_info=True)
            df_out['VWAP'] = np.nan
    else:
        logger.warning("VWAP calculation requires 'high', 'low', 'close', and 'volume' columns. Skipping VWAP.")
        df_out['VWAP'] = np.nan

    # 3. Volume EMA - 20 period for volume
    if 'volume' in df_out.columns:
        try:
            df_out['Volume_EMA_20'] = df_out['volume'].ewm(span=20, adjust=False, min_periods=20).mean()
            logger.info("Calculated Volume_EMA_20.")
        except Exception as e:
            logger.error(f"Error calculating Volume_EMA_20: {e}", exc_info=True)
            df_out['Volume_EMA_20'] = np.nan
    else:
        logger.warning("'volume' column not found, skipping Volume_EMA_20.")
        df_out['Volume_EMA_20'] = np.nan

    # 4. RSI (Relative Strength Index) - 14 period for close price
    if 'close' in df_out.columns:
        try:
            delta = df_out['close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df_out['RSI_14'] = rsi
            logger.info("Calculated RSI_14.")
        except Exception as e:
            logger.error(f"Error calculating RSI_14: {e}", exc_info=True)
            df_out['RSI_14'] = np.nan
    else:
        logger.warning("'close' column not found, skipping RSI_14.")
        df_out['RSI_14'] = np.nan

    # Ensure original columns are present even if calculations fail for some indicators
    for col in ['EMA_200', 'VWAP', 'Volume_EMA_20', 'RSI_14']:
        if col not in df_out:
            df_out[col] = np.nan
            logger.info(f"Column {col} was not created, adding as NaN.")

    return df_out

# --- Plotting Functions ---
def plot_data_with_indicators(df, timeframe_name, contract_sym, save_image=False, num_bars=NUMBER_OF_BARS_TO_PLOT):
    """Plot financial data with candlestick chart and technical indicators using mplfinance."""
    if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        logger.warning(f"Missing required OHLC columns to plot for {timeframe_name}")
        return None
    
    # Create a copy of the dataframe with the specified number of bars
    df_plot = df.tail(num_bars).copy()
    
    # Ensure index is datetime
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        logger.warning(f"Converting index to DatetimeIndex for {timeframe_name}")
        df_plot.index = pd.to_datetime(df_plot.index)
    
    # Prepare plot data
    title = f'{contract_sym} - {timeframe_name} Chart ({datetime.now().strftime("%Y-%m-%d %H:%M")})'
    
    # Define style
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge={'up':'green', 'down':'red'},
        wick={'up':'green', 'down':'red'},
        volume={'up':'green', 'down':'red'},
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridstyle='--',
        gridaxis='both',
        rc={'axes.grid': True, 'axes.grid.axis': 'both'},
        y_on_right=False,
    )
    
    # Prepare additional plots
    plots = []
    
    # Add VWAP if available
    if 'VWAP' in df_plot.columns:
        vwap_plot = mpf.make_addplot(df_plot['VWAP'], color='purple', width=1.5, label='VWAP')
        plots.append(vwap_plot)
    
    # Add 200 EMA if available
    if 'EMA_200' in df_plot.columns:
        ema_plot = mpf.make_addplot(df_plot['EMA_200'], color='orange', width=1.5, linestyle='--', label='200 EMA')
        plots.append(ema_plot)
    
    # Add RSI if available
    if 'RSI_14' in df_plot.columns:
        rsi_plot = mpf.make_addplot(df_plot['RSI_14'], panel=2, color='blue', width=1, 
                              ylim=(0, 100), ylabel='RSI')
        plots.append(rsi_plot)
        
        # Add RSI overbought line (70)
        overbought_plot = mpf.make_addplot([70] * len(df_plot), panel=2, color='red', 
                                    width=0.8, linestyle='--')
        plots.append(overbought_plot)
        
        # Add RSI midline (50)
        midline_plot = mpf.make_addplot([50] * len(df_plot), panel=2, color='gray', 
                                width=0.5, linestyle=':')
        plots.append(midline_plot)
        
        # Add RSI oversold line (30)
        oversold_plot = mpf.make_addplot([30] * len(df_plot), panel=2, color='green', 
                                  width=0.8, linestyle='--')
        plots.append(oversold_plot)
    
    # Add Volume EMA if available
    if 'Volume_EMA_20' in df_plot.columns and 'volume' in df_plot.columns:
        vol_ema_plot = mpf.make_addplot(df_plot['Volume_EMA_20'], panel=1, color='blue', 
                                 width=1, ylabel='Volume')
        plots.append(vol_ema_plot)
    
    # Configure panel ratios and spacing
    panel_ratios = (4, 1, 1) if 'RSI_14' in df_plot.columns else (4, 1)
    
    # Setup figure and save path
    image_path = None
    if save_image:
        safe_contract_sym = "".join(c if c.isalnum() else "_" for c in contract_sym)
        image_filename = f"{safe_contract_sym}_{timeframe_name}_chart.png"
        image_path = os.path.join(PLOTS_DIR, image_filename)
    
    # Create the plot
    try:
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=s,
            title=title,
            figsize=(14, 16),
            volume=True,
            volume_panel=1,
            panel_ratios=panel_ratios,
            addplot=plots,
            returnfig=True,
            savefig=image_path if save_image else None,
            tight_layout=True,
            show_nontrading=False
        )
        
        # Add legend
        legend_elements = []
        if 'VWAP' in df_plot.columns:
            legend_elements.append(plt.Line2D([0], [0], color='purple', lw=1.5, label='VWAP'))
        if 'EMA_200' in df_plot.columns:
            legend_elements.append(plt.Line2D([0], [0], color='orange', lw=1.5, linestyle='--', label='200 EMA'))
        
        if legend_elements:
            axes[0].legend(handles=legend_elements, loc='upper left')
        
        # Show RSI legend on its panel if RSI is plotted
        if 'RSI_14' in df_plot.columns:
            rsi_elements = [
                plt.Line2D([0], [0], color='blue', lw=1, label='RSI (14)'),
                plt.Line2D([0], [0], color='red', lw=0.8, linestyle='--', label='Overbought (70)'),
                plt.Line2D([0], [0], color='green', lw=0.8, linestyle='--', label='Oversold (30)')
            ]
            axes[len(axes)-1].legend(handles=rsi_elements, loc='upper left')
        
        if not save_image:
            plt.show()
        else:
            logger.info(f"Chart saved to {image_path}")
            fig.clf()
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error creating chart for {timeframe_name}: {e}", exc_info=True)
        if save_image:
            image_path = None
    
    return image_path

def create_composite_chart(chart_image_paths_dict, contract_sym):
    """Combines multiple timeframe charts into a single image."""
    if not chart_image_paths_dict:
        logger.warning("No chart images to combine")
        return None
        
    try:
        # Define the order of timeframes to display
        ordered_timeframes = ['5min','15min', '30min', '1hour', '4hour']
        valid_images = [tf for tf in ordered_timeframes if tf in chart_image_paths_dict]
        
        if not valid_images:
            logger.warning("No valid timeframe images found")
            return None
            
        # Open all images and get dimensions
        images = []
        for tf in valid_images:
            if tf in chart_image_paths_dict and os.path.exists(chart_image_paths_dict[tf]):
                img = Image.open(chart_image_paths_dict[tf])
                images.append((tf, img))
            
        if not images:
            logger.warning("Could not open any chart images")
            return None
            
        # Calculate composite image dimensions
        img_width = images[0][1].width
        img_height = images[0][1].height
        
        # Create grid layout based on number of images
        if len(images) <= 2:
            grid_cols = 1
            grid_rows = len(images)
        else:
            grid_cols = 2
            grid_rows = (len(images) + 1) // 2
            
        # Create a new image with the calculated dimensions
        composite_width = grid_cols * img_width
        composite_height = grid_rows * img_height + 50  # Extra space for title
        
        # Create new image with white background
        composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
        draw = ImageDraw.Draw(composite)
        
        # Add title
        title = f"{contract_sym} - Multi-Timeframe Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        try:
            # Try to use a nice font if available
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            
        # Draw title at the top center
        title_width = draw.textlength(title, font=font) if hasattr(draw, 'textlength') else len(title) * 12
        draw.text(((composite_width - title_width) // 2, 10), title, fill=(0, 0, 0), font=font)
        
        # Paste each image into the composite
        for i, (tf, img) in enumerate(images):
            row = i // grid_cols
            col = i % grid_cols
            x_offset = col * img_width
            y_offset = row * img_height + 50  # Account for title space
            
            # Add timeframe label
            draw.text((x_offset + 10, y_offset + 10), tf, fill=(0, 0, 0), font=font)
            
            # Paste the image
            composite.paste(img, (x_offset, y_offset))
            
        # Save the composite image
        composite_path = os.path.join(PLOTS_DIR, f"{contract_sym}_composite_chart.png")
        composite.save(composite_path, dpi=(150, 150))
        
        # Compress the image to reduce file size
        try:
            # Open the saved image
            img = Image.open(composite_path)
            # Compress and save with lower quality
            img.save(composite_path, optimize=True, quality=85)  # Adjust quality as needed (lower = smaller file)
            logger.info(f"Compressed composite image to reduce file size")
        except Exception as e:
            logger.error(f"Error compressing composite image: {e}")
            
        logger.info(f"Created composite chart at {composite_path}")
        return composite_path
        
    except Exception as e:
        logger.error(f"Error creating composite chart: {e}", exc_info=True)
        return None

# --- Gemini Analysis Functions ---
def analyze_composite_chart_with_gemini(composite_image_path, contract_symbol_for_prompt):
    """Analyze a composite chart containing multiple timeframes using Gemini."""
    global gemini_model, api_call_times
    if not gemini_model:
        logger.error("Gemini model not initialized for composite chart analysis.")
        return None
        
    if not composite_image_path or not os.path.exists(composite_image_path):
        logger.warning("Invalid composite chart path provided to Gemini.")
        return None
        
    logger.info(f"Analyzing composite chart for {contract_symbol_for_prompt} using Gemini...")
    
    # Track this call for rate limiting analysis
    now = time.time()
    api_call_times.append(now)
    
    # Log API call frequency
    recent_calls = [t for t in api_call_times if now - t < 60]  # Calls in last minute
    logger.info(f"Making Gemini API call. This is call #{len(recent_calls)} in the last 60 seconds")
    
    # Clean up old entries
    api_call_times = [t for t in api_call_times if now - t < 3600]  # Keep last hour
    
    # Implement retry with exponential backoff
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            # Prepare the image for Gemini
            img = Image.open(composite_image_path)
            
            # Prepare the prompt
            prompt_parts = [
                f"""You are an expert multi-timeframe technical analyst and trader expert on price action as well as ICT concept. For each trade, let's use 5min graph or 15min graph to determine the target profit and stop loss especially care about the support and resistance level.
let's use relatively balance profit target and stop loss so that we may not too be aggressive.
I'm providing you with a composite chart image for {contract_symbol_for_prompt} that contains multiple timeframes (5min,15min, 30min, 1hour, 4hour).
Each timeframe is labeled directly on the chart. Please analyze ALL timeframes visible in this composite image.

Based on a holistic analysis of ALL timeframes visible in this image:
1.  Dominant Trend Direction: (e.g., Strong Bullish, Weak Bullish, Range/Neutral, Weak Bearish, Strong Bearish).
2.  Overall Setup Quality: Grade from C to A+. If None, state "None". Justify.
3.  Trade Direction: (Long / Short / None). If None, skip points 4-7.
4.  Suggested Entry Price: (Specific approx price or "Market" or condition like "breakout above X on 15min").
5.  Suggested Stop-Loss Price: (Specific approx price).
6.  Suggested Profit Target Price: (Specific approx price).
7.  Confidence Level for this Trade: (Low / Medium / High).
8.  Brief Rationale: Key confluences/divergences leading to this decision.

Output in a clear, structured format using these numbered points. Example for long:
Trade Direction: Long
Suggested Entry Price: Market
Suggested Stop-Loss Price: 1230.50
Suggested Profit Target Price: 1255.00
Confidence Level for this Trade: Medium
Rationale: Strong uptrend on 4hr/1hr. 15min pullback to VWAP.
""",
                img
            ]
            
            # Send to Gemini
            logger.info(f"Sending composite image to Gemini (attempt {retry_count+1}/{max_retries})...")
            response = gemini_model.generate_content(prompt_parts)
            logger.info(f"--- Gemini Composite Chart Analysis for {contract_symbol_for_prompt} ---\n{response.text}\n-----------------------------------------------------")
            return response.text
            
        except Exception as e:
            if "429" in str(e):
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                logger.error(f"Error during Gemini API composite chart call: {e}", exc_info=True)
                if hasattr(e, 'response') and e.response:
                    logger.error(f"Gemini API Error Details: {e.response}")
                return None
    
    logger.error(f"Failed to get Gemini response after {max_retries} retries")
    return None

def parse_gemini_multi_chart_analysis(analysis_text):
    """Parse Gemini's analysis text, handling markdown formatting like ** for bold text."""
    parsed = {
        "trade_direction": "None", 
        "entry_price_desc": "Market", 
        "stop_loss": None, 
        "profit_target": None, 
        "confidence": "Low", 
        "setup_quality": "C", 
        "rationale": "Not specified.", 
        "raw_text": analysis_text
    }
    
    if not analysis_text:
        return parsed
        
    # Clean up the text by removing markdown formatting
    def clean_markdown(text):
        # Remove ** bold markers
        return text.replace('**', '')
    
    lines = analysis_text.split('\n')
    for line in lines:
        line = clean_markdown(line.strip())
        parts = line.split(":", 1)
        
        if len(parts) < 2:
            continue
            
        key = parts[0].strip().lower()
        value = parts[1].strip()
        
        # Parse trade direction
        if "trade direction" in key:
            value_lower = value.lower()
            if "long" in value_lower or "buy" in value_lower:
                parsed["trade_direction"] = "Long"
            elif "short" in value_lower or "sell" in value_lower:
                parsed["trade_direction"] = "Short"
            elif "none" in value_lower:
                parsed["trade_direction"] = "None"
                
        # Parse entry price
        elif "suggested entry price" in key or "entry price" in key:
            parsed["entry_price_desc"] = value
            # Try to extract numeric price if present
            try:
                import re
                # Look for numeric values with optional commas
                price_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', value)
                if price_match:
                    numeric_price = price_match.group(1).replace(',', '')
                    parsed["entry_price"] = float(numeric_price)
            except Exception as e:
                logger.warning(f"Could not extract numeric entry price from: {value}. Error: {e}")
                
        # Parse stop-loss price
        elif "suggested stop-loss price" in key or "stop-loss" in key or "stop loss" in key:
            try:
                # Clean and extract the numeric part
                clean_value = value.replace(',', '')
                import re
                # Find the first number in the string
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', clean_value)
                if numeric_match:
                    parsed["stop_loss"] = float(numeric_match.group(1))
                    parsed["stop_loss_desc"] = value
                else:
                    logger.warning(f"Could not find numeric stop loss in: {value}")
                    parsed["stop_loss_desc"] = value
            except Exception as e:
                logger.warning(f"Could not parse SL price: {value}. Error: {e}")
                parsed["stop_loss_desc"] = value
                
        # Parse profit target price
        elif "suggested profit target price" in key or "profit target" in key or "take profit" in key or "take-profit" in key:
            try:
                # Clean and extract the numeric part
                clean_value = value.replace(',', '')
                import re
                # Find the first number in the string
                numeric_match = re.search(r'(\d+(?:\.\d+)?)', clean_value)
                if numeric_match:
                    parsed["profit_target"] = float(numeric_match.group(1))
                    parsed["profit_target_desc"] = value
                else:
                    logger.warning(f"Could not find numeric profit target in: {value}")
                    parsed["profit_target_desc"] = value
            except Exception as e:
                logger.warning(f"Could not parse TP price: {value}. Error: {e}")
                parsed["profit_target_desc"] = value
                
        # Parse confidence level
        elif "confidence level" in key or "confidence" in key:
            value_lower = value.lower()
            if "high" in value_lower:
                parsed["confidence"] = "High"
            elif "medium" in value_lower:
                parsed["confidence"] = "Medium"
            # Low is the default
                
        # Parse setup quality
        elif "setup quality" in key or "quality" in key:
            quality_val = value.lower()
            # Extract letter grade
            if "a+" in quality_val:
                parsed["setup_quality"] = "A+"
            elif "a" in quality_val and "a+" not in quality_val:
                parsed["setup_quality"] = "A"
            elif "b+" in quality_val:
                parsed["setup_quality"] = "B+"
            elif "b" in quality_val and "b+" not in quality_val:
                parsed["setup_quality"] = "B"
            elif "c+" in quality_val:
                parsed["setup_quality"] = "C+"
            elif "c" in quality_val and "c+" not in quality_val:
                parsed["setup_quality"] = "C"
                
        # Parse rationale
        elif "rationale" in key or "brief rationale" in key:
            parsed["rationale"] = value
    
    logger.info(f"Parsed Gemini Decision: Direction={parsed['trade_direction']}, "
                f"Entry='{parsed['entry_price_desc']}', SL={parsed['stop_loss']}, "
                f"TP={parsed['profit_target']}, Confidence={parsed['confidence']}, "
                f"Quality={parsed['setup_quality']}")
    return parsed

# --- Risk Management Functions ---
def get_current_total_position_size():
    """
    Get the current total number of contracts across all open positions.
    
    Returns:
    --------
    int: Total number of contracts currently held across all positions
    """
    global futuresdesk_client, ACCOUNT_ID
    
    if not futuresdesk_client or not ACCOUNT_ID:
        logger.warning("Client or account not initialized for position check")
        return 0
    
    try:
        positions = futuresdesk_client.position.search_open_positions(account_id=ACCOUNT_ID)
        total_contracts = 0
        
        for pos in positions:
            position_size = abs(pos.get('size', 0))  # Use absolute value to count both long and short
            total_contracts += position_size
            logger.debug(f"Position {pos.get('contract_id', 'Unknown')}: {position_size} contracts")
        
        logger.info(f"Current total position size: {total_contracts} contracts")
        return total_contracts
        
    except Exception as e:
        logger.error(f"Error getting current position size: {e}")
        return 0

def calculate_position_size_with_global_limit(entry_price, stop_loss, max_risk_per_trade=MAX_RISK_PER_TRADE, 
                                            max_contracts=MAX_CONTRACTS, max_total_contracts=MAX_TOTAL_CONTRACTS,
                                            contract_multiplier=None, tick_size=None, tick_value=None):
    """
    Calculate the appropriate position size based on risk management rules and global position limits.
    
    Parameters:
    -----------
    entry_price : float
        The entry price for the trade
    stop_loss : float
        The stop loss price for the trade
    max_risk_per_trade : float
        Maximum dollar amount to risk per trade
    max_contracts : int
        Maximum number of contracts allowed per individual trade
    max_total_contracts : int
        Maximum total contracts allowed across all positions
    contract_multiplier : float
        Contract multiplier (specific to the instrument)
    tick_size : float
        Minimum price movement (tick size)
    tick_value : float
        Dollar value of one tick
    
    Returns:
    --------
    tuple: (position_size, can_trade, reason)
        position_size: Number of contracts to trade (0 if can't trade)
        can_trade: Boolean indicating if trade can proceed
        reason: String explaining why trade was limited or rejected
    """
    # Get current total position size
    current_total_contracts = get_current_total_position_size()
    
    # Check if we're already at or above the global limit
    if current_total_contracts >= max_total_contracts:
        reason = f"Cannot trade: Already at maximum total position limit ({current_total_contracts}/{max_total_contracts} contracts)"
        logger.warning(reason)
        return 0, False, reason
    
    # Calculate available contract capacity
    available_contracts = max_total_contracts - current_total_contracts
    logger.info(f"Available contract capacity: {available_contracts} contracts (Current: {current_total_contracts}, Max: {max_total_contracts})")
    
    # Contract-specific values if not provided
    if contract_multiplier is None:
        contract_multiplier = 2  # MNQ default
    
    if tick_size is None:
        tick_size = 0.25  # MNQ default
    
    if tick_value is None:
        tick_value = tick_size * contract_multiplier
    
    # Calculate risk-based position size (original logic)
    risk_points = abs(entry_price - stop_loss)
    risk_ticks = risk_points / tick_size
    dollar_risk_per_contract = risk_ticks * tick_value
    
    if dollar_risk_per_contract <= 0:
        logger.warning("Risk per contract is zero or negative. Using minimum position size.")
        risk_based_size = 1
    else:
        risk_based_size = int(max_risk_per_trade / dollar_risk_per_contract)
    
    # Apply individual trade contract limit
    individual_limit_size = min(risk_based_size, max_contracts)
    
    # Apply global position limit
    global_limit_size = min(individual_limit_size, available_contracts)
    
    # Minimum position size is 1, but only if we have capacity
    if global_limit_size <= 0:
        reason = f"Cannot trade: No available capacity (need 1, have {available_contracts} available)"
        logger.warning(reason)
        return 0, False, reason
    
    final_position_size = max(global_limit_size, 1)
    
    # Final check to ensure we don't exceed global limit
    if current_total_contracts + final_position_size > max_total_contracts:
        final_position_size = max_total_contracts - current_total_contracts
        if final_position_size <= 0:
            reason = f"Cannot trade: Adding {final_position_size} contracts would exceed global limit"
            logger.warning(reason)
            return 0, False, reason
    
    # Generate informative reason
    limitations = []
    if final_position_size < risk_based_size:
        if final_position_size == max_contracts:
            limitations.append(f"limited by individual trade max ({max_contracts})")
        if final_position_size == available_contracts:
            limitations.append(f"limited by global position limit ({available_contracts} available)")
    
    reason = f"Position size: {final_position_size} contracts"
    if limitations:
        reason += f" ({', '.join(limitations)})"
    
    logger.info(f"Risk calculation: Entry: {entry_price}, SL: {stop_loss}, " 
                f"Risk: {risk_points} points (${dollar_risk_per_contract:.2f} per contract), "
                f"Risk-based size: {risk_based_size}, Individual limit: {individual_limit_size}, "
                f"Global limit: {global_limit_size}, Final size: {final_position_size} contracts, "
                f"Total risk: ${dollar_risk_per_contract * final_position_size:.2f}")
    
    return final_position_size, True, reason

def calculate_position_size(entry_price, stop_loss, max_risk_per_trade=MAX_RISK_PER_TRADE, max_contracts=MAX_CONTRACTS, contract_multiplier=None, tick_size=None, tick_value=None):
    """
    Legacy function maintained for compatibility. 
    Now calls the enhanced version with global limits.
    """
    position_size, can_trade, reason = calculate_position_size_with_global_limit(
        entry_price, stop_loss, max_risk_per_trade, max_contracts, MAX_TOTAL_CONTRACTS,
        contract_multiplier, tick_size, tick_value
    )
    
    if not can_trade:
        logger.warning(f"Position sizing rejected: {reason}")
        return 0
    
    return position_size

# --- Trading Functions with OCO Support ---
def place_trade_with_oco(contract_id_to_trade, account_id_to_trade, side, entry_price, stop_loss_price, 
                        take_profit_price, order_type_code=2, limit_price=None, max_risk=MAX_RISK_PER_TRADE, max_contracts=MAX_CONTRACTS):
    """Places a trade with FuturesDesk API including stop-loss and take-profit orders with OCO management."""
    global futuresdesk_client, CONTRACT_SYMBOL, oco_manager
    if not futuresdesk_client: 
        logger.error("FuturesDesk client not initialized for place_trade.")
        return None
        
    # Convert side to the correct code
    actual_side_code = None
    if side.upper() == "BUY" or side.upper() == "LONG": 
        actual_side_code = 0  # 0 = Bid (buy)
        opposite_side_code = 1  # 1 = Ask (sell) - for SL/TP orders
        trade_side = "LONG"
    elif side.upper() == "SELL" or side.upper() == "SHORT": 
        actual_side_code = 1  # 1 = Ask (sell)
        opposite_side_code = 0  # 0 = Bid (buy) - for SL/TP orders
        trade_side = "SHORT"
    else: 
        logger.error(f"Invalid trade side: {side}")
        return None
        
    if actual_side_code is None: 
        logger.error("Side code error. Aborting trade.")
        return None
        
    # Validate that stop loss and take profit are in correct relation to entry
    valid_trade = True
    if actual_side_code == 0:  # LONG
        if stop_loss_price >= entry_price:
            logger.error(f"LONG SL validation fail: SL {stop_loss_price} >= Entry {entry_price}")
            valid_trade = False
        if take_profit_price <= entry_price:
            logger.error(f"LONG TP validation fail: TP {take_profit_price} <= Entry {entry_price}")
            valid_trade = False
    else:  # SHORT
        if stop_loss_price <= entry_price:
            logger.error(f"SHORT SL validation fail: SL {stop_loss_price} <= Entry {entry_price}")
            valid_trade = False
        if take_profit_price >= entry_price:
            logger.error(f"SHORT TP validation fail: TP {take_profit_price} >= Entry {entry_price}")
            valid_trade = False
            
    if not valid_trade:
        logger.error("Trade aborted due to SL/TP validation failure.")
        return None
        
    # Calculate position size based on risk management rules with global limit check
    contract_multiplier, tick_size, tick_value = get_instrument_details(CONTRACT_SYMBOL)
    position_size, can_trade, sizing_reason = calculate_position_size_with_global_limit(
        entry_price=entry_price,
        stop_loss=stop_loss_price,
        max_risk_per_trade=max_risk,
        max_contracts=max_contracts,
        max_total_contracts=MAX_TOTAL_CONTRACTS,
        contract_multiplier=contract_multiplier,
        tick_size=tick_size,
        tick_value=tick_value
    )
    
    # Check if we can proceed with the trade
    if not can_trade:
        logger.error(f"Trade rejected due to position limits: {sizing_reason}")
        return None
    
    logger.info(f"Position sizing approved: {sizing_reason} for {side} at {entry_price}")
        
    try:
        # First check for existing positions
        existing_positions = futuresdesk_client.position.search_open_positions(account_id=account_id_to_trade)
        existing_position = None
        for pos in existing_positions:
            if pos.get('contract_id') == contract_id_to_trade:
                existing_position = pos
                break
                
        # If there's an existing position in the opposite direction, close it
        if existing_position:
            existing_side = "LONG" if existing_position.get('side') == 0 else "SHORT"
            if (side.upper() == "BUY" and existing_side == "SHORT") or (side.upper() == "SELL" and existing_side == "LONG"):
                logger.info(f"Closing existing {existing_side} position before opening new {side} position")
                # Close existing position
                futuresdesk_client.position.close_position(
                    account_id=account_id_to_trade, 
                    contract_id=contract_id_to_trade
                )
                logger.info(f"Closed existing {existing_side} position for {contract_id_to_trade}")
            else:
                # Same direction, might want to add to position or do nothing
                logger.info(f"Existing {existing_side} position matches new {side} signal. Not closing.")
                
        # Cancel any open orders for this contract
        open_orders = futuresdesk_client.order.search_open_orders(account_id=account_id_to_trade)
        for order in open_orders:
            if order.get('contract_id') == contract_id_to_trade:
                logger.info(f"Cancelling existing order ID: {order.get('id')}")
                futuresdesk_client.order.cancel_order(
                    account_id=account_id_to_trade,
                    order_id=order.get('id')
                )
                
        # Now place the main order
        logger.info(f"Placing {side} order: Type={order_type_code}, Size={position_size}, Contract={contract_id_to_trade}")
        
        # Place the main order
        main_order_params = {
            'account_id': account_id_to_trade,
            'contract_id': contract_id_to_trade,
            'type': order_type_code,  # 2 = Market by default
            'side': actual_side_code,
            'size': position_size  # Using risk-based position size
        }
        
        # Add optional parameters if provided
        if limit_price is not None:
            main_order_params['limit_price'] = limit_price
            
        # Place the main order
        main_order_id = futuresdesk_client.order.place_order(**main_order_params)
        
        logger.info(f"Main order placed successfully. Order ID: {main_order_id}")
        
        # Initialize OCO tracking variables
        sl_order_id = None
        tp_order_id = None
        
        # Place stop loss order if provided
        if stop_loss_price:
            sl_order_type = 4  # Stop order
            sl_order_id = futuresdesk_client.order.place_order(
                account_id=account_id_to_trade,
                contract_id=contract_id_to_trade,
                type=sl_order_type,
                side=opposite_side_code,  # Opposite side of the main order
                size=position_size,  # Using risk-based position size
                stop_price=stop_loss_price
            )
            logger.info(f"Stop-loss order placed. Order ID: {sl_order_id}, Price: {stop_loss_price}")
            
        # Place take profit order if provided
        if take_profit_price:
            tp_order_type = 1  # Limit order
            tp_order_id = futuresdesk_client.order.place_order(
                account_id=account_id_to_trade,
                contract_id=contract_id_to_trade,
                type=tp_order_type,
                side=opposite_side_code,  # Opposite side of the main order
                size=position_size,  # Using risk-based position size
                limit_price=take_profit_price
            )
            logger.info(f"Take-profit order placed. Order ID: {tp_order_id}, Price: {take_profit_price}")
        
        # Create OCO group for tracking
        if oco_manager and (sl_order_id or tp_order_id):
            position_id = f"{contract_id_to_trade}_{trade_side}_{int(time.time())}"
            oco_manager.create_oco_group(
                position_id=position_id,
                main_order_id=main_order_id,
                stop_loss_order_id=sl_order_id,
                take_profit_order_id=tp_order_id,
                contract_id=contract_id_to_trade,
                trade_side=trade_side,
                entry_price=entry_price
            )
            logger.info(f"Created OCO group {position_id} for trade management")
            
        return main_order_id
        
    except Exception as e:
        logger.error(f"Error placing order: {e}", exc_info=True)
        return None

def manage_active_trades():
    """Monitor and manage active trades including OCO orders and global position limits."""
    global oco_manager
    
    if not oco_manager:
        logger.warning("OCO manager not initialized, skipping trade management")
        return
    
    try:
        # Check current total position size
        current_total_contracts = get_current_total_position_size()
        active_groups = oco_manager.get_active_oco_groups()
        
        if not active_groups:
            logger.debug("No active OCO groups to manage")
            if current_total_contracts > 0:
                logger.info(f"Current total position: {current_total_contracts} contracts (no OCO tracking)")
            return
            
        logger.info(f"Managing {len(active_groups)} active OCO groups. Total position: {current_total_contracts}/{MAX_TOTAL_CONTRACTS} contracts")
        
        # Check for global position limit violations
        if current_total_contracts > MAX_TOTAL_CONTRACTS:
            logger.warning(f"WARNING: Total position ({current_total_contracts}) exceeds maximum allowed ({MAX_TOTAL_CONTRACTS})!")
            # Could implement emergency position reduction here if needed
        
        for position_id, oco_group in active_groups.items():
            try:
                # Log current status
                logger.info(f"OCO Group {position_id}: "
                           f"Main filled: {oco_group['main_order_filled']}, "
                           f"Position active: {oco_group['position_active']}, "
                           f"Size: {oco_group.get('position_size', 0)} contracts, "
                           f"Unrealized P&L: ${oco_group.get('unrealized_pnl', 0):.2f}")
                
                # Additional management logic can be added here
                # For example: trailing stops, position sizing adjustments, etc.
                
            except Exception as e:
                logger.error(f"Error managing OCO group {position_id}: {e}")
                
    except Exception as e:
        logger.error(f"Error in manage_active_trades: {e}", exc_info=True)

def make_final_trade_from_gemini_decision(gemini_decision, current_market_data_dict):
    """Execute trades based on Gemini's analysis with OCO management."""
    global ACCOUNT_ID, CONTRACT_ID, CONTRACT_DISPLAY_SYMBOL
    logger.info("--- Acting on Gemini's Unified Trading Decision ---")
    
    # Check if we have a valid decision
    if not gemini_decision or gemini_decision["trade_direction"] == "None":
        logger.info("Gemini recommends no trade or decision invalid.")
        return
    
    # Filter for high confidence or at least B+ setup quality
    confidence_level = gemini_decision["confidence"]
    setup_quality = gemini_decision["setup_quality"]
    
    # Check if setup meets our criteria
    if confidence_level != "High" and setup_quality not in ["A+", "A", "B+"]:
        logger.info(f"Trade doesn't meet quality criteria. Confidence: {confidence_level}, Setup Quality: {setup_quality}")
        logger.info("Only executing trades with High confidence or at least B+ setup quality.")
        return
        
    # Extract decision details
    trade_action = gemini_decision["trade_direction"].upper()
    entry_desc = gemini_decision["entry_price_desc"]
    stop_loss = gemini_decision["stop_loss"]
    take_profit = gemini_decision["profit_target"]
    
    logger.info(f"Gemini Decision: {trade_action} {CONTRACT_DISPLAY_SYMBOL} | Quality: {gemini_decision['setup_quality']} | Confidence: {gemini_decision['confidence']}")
    logger.info(f"Entry: {entry_desc} | SL: {stop_loss} | TP: {take_profit} | Rationale: {gemini_decision.get('rationale', 'N/A')}")
    
    # Validate stop loss and take profit
    if not stop_loss or not take_profit:
        logger.critical("SL or TP missing. Aborting trade.")
        return
        
    # Get latest price from market data for validation
    latest_close = None
    for tf_key in ["1min", "5min", "15min"]:
        if tf_key in current_market_data_dict and not current_market_data_dict[tf_key].empty:
            latest_close = current_market_data_dict[tf_key]['close'].iloc[-1]
            logger.info(f"Validation price from {tf_key}: {latest_close}")
            break
            
    if not latest_close:
        logger.error("No latest market price for validation. Aborting trade.")
        return
    
    # Determine entry type (market or limit) and price
    order_type = 2  # Default to market order (2)
    limit_price = None
    entry_price = latest_close  # Default entry price for market orders is current price
    
    # If entry price is specified and not "market", use a limit order
    if not isinstance(entry_desc, str) or "market" not in entry_desc.lower():
        try:
            # Try to parse as a number
            if isinstance(entry_desc, (int, float)):
                parsed_entry = float(entry_desc)
                entry_price = parsed_entry  # Update entry price for position sizing
                
                # If entry price is close to current price (within 0.5%), use market order
                if abs(parsed_entry - latest_close) < (latest_close * 0.005):
                    logger.info(f"Entry price {parsed_entry} is close to current price {latest_close}. Using market order.")
                    order_type = 2  # Market
                else:
                    logger.info(f"Entry price {parsed_entry} specified. Using limit order.")
                    order_type = 1  # Limit
                    limit_price = parsed_entry
            else:
                # Conditional entry that can't be automated, use market order but log the condition
                logger.info(f"Conditional entry: '{entry_desc}'. Using market order but might need manual intervention.")
                order_type = 2  # Market
        except (ValueError, TypeError):
            logger.warning(f"Could not parse entry price: '{entry_desc}'. Using market order.")
            order_type = 2  # Market
    
    # Execute the trade with OCO management
    logger.info(f"Proceeding with {trade_action} order for {CONTRACT_DISPLAY_SYMBOL} with order type {order_type}")
    
    # Use the updated place_trade function with OCO support
    order_id = place_trade_with_oco(
        contract_id_to_trade=CONTRACT_ID,
        account_id_to_trade=ACCOUNT_ID,
        side=trade_action,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        order_type_code=order_type,
        limit_price=limit_price,
        max_risk=MAX_RISK_PER_TRADE,
        max_contracts=MAX_CONTRACTS
    )
    
    if order_id:
        logger.info(f"Successfully placed {trade_action} order with OCO management. Order ID: {order_id}")
    else:
        logger.error("Failed to place order.")
    
    logger.info(f"Order Type: {'Market' if order_type == 2 else 'Limit'}")
    if limit_price:
        logger.info(f"Limit Price: {limit_price}")
    logger.info(f"Stop Loss: {stop_loss}")
    logger.info(f"Take Profit: {take_profit}")
    logger.info(f"Confidence: {confidence_level} | Setup Quality: {setup_quality}")
    logger.info("****************************************************************")

# --- Main Trading Logic ---
def trading_bot_cycle_content():
    """The actual content of a single trading cycle."""
    global all_data

    logger.info(f"--- Starting Trading Bot Cycle Content: {datetime.now()} ---")

    if not CONTRACT_ID or not ACCOUNT_ID:
        logger.error("Contract ID or Account ID not set. Halting cycle content.")
        return

    timeframes_for_gemini_prompt = {
        "5min": {"unit": 1, "unit_number": 5, "days": DAYS_OF_DATA_TO_FETCH, "bars": BAR_LIMIT_PER_FETCH},
        "15min": {"unit": 1, "unit_number": 15, "days": DAYS_OF_DATA_TO_FETCH, "bars": BAR_LIMIT_PER_FETCH},
        "30min": {"unit": 1, "unit_number": 30, "days": DAYS_OF_DATA_TO_FETCH, "bars": BAR_LIMIT_PER_FETCH},
        "1hour": {"unit": 3, "unit_number": 1, "days": DAYS_OF_DATA_TO_FETCH, "bars": BAR_LIMIT_PER_FETCH},
        "4hour": {"unit": 3, "unit_number": 4, "days": DAYS_OF_DATA_TO_FETCH + 30, "bars": BAR_LIMIT_PER_FETCH},
    }
    other_timeframes_to_fetch = {
         "1min": {"unit": 1, "unit_number": 1, "days": 2, "bars": BAR_LIMIT_PER_FETCH},
    }
    all_timeframes_config = {**timeframes_for_gemini_prompt, **other_timeframes_to_fetch}

    all_data = {}
    saved_chart_paths_for_gemini = {}

    # 1. Fetch Data and Calculate Indicators
    logger.info("Fetching data and calculating indicators...")
    critical_fetch_ok = True
    for tf_name, tf_params in all_timeframes_config.items():
        df = fetch_historical_data(CONTRACT_ID, tf_params["unit"], tf_params["unit_number"],
                                   days_back=tf_params["days"], limit_bars=tf_params["bars"])
        if not df.empty:
            all_data[tf_name] = calculate_indicators(df.copy())
        else:
            all_data[tf_name] = pd.DataFrame()
            logger.warning(f"No data for {tf_name}.")
            if tf_name in timeframes_for_gemini_prompt:
                critical_fetch_ok = False

    if not critical_fetch_ok:
        logger.error("Failed to fetch data for one or more critical timeframes needed for Gemini. Skipping Gemini analysis for this cycle.")
        return

    # 2. Manage existing positions (OCO and other management)
    logger.info("Managing existing positions...")
    manage_active_trades()

    # 3. Check if we already have positions in this contract
    has_existing_positions = False
    try:
        existing_positions = futuresdesk_client.position.search_open_positions(account_id=ACCOUNT_ID)
        for pos in existing_positions:
            if pos.get('contract_id') == CONTRACT_ID:
                has_existing_positions = True
                logger.info(f"Active position detected for {CONTRACT_DISPLAY_SYMBOL}. Will skip new trade signals for this cycle.")
                break
    except Exception as e:
        logger.error(f"Error checking existing positions: {e}", exc_info=True)
    
    # Only look for new trade setups if we don't have active positions
    if not has_existing_positions:
        logger.info("No active position. Plotting individual charts for new trade analysis...")
        charts_generated_count = 0
        if CONTRACT_ID and CONTRACT_DISPLAY_SYMBOL:
            for tf_name in timeframes_for_gemini_prompt.keys():
                if tf_name in all_data and not all_data[tf_name].empty:
                    img_path = plot_data_with_indicators(all_data[tf_name], tf_name, CONTRACT_DISPLAY_SYMBOL, save_image=True)
                    if img_path:
                        saved_chart_paths_for_gemini[tf_name] = img_path
                        charts_generated_count += 1
                else:
                    logger.warning(f"Cannot plot chart for Gemini ({tf_name}), data missing.")
        
        if charts_generated_count < len(timeframes_for_gemini_prompt):
            logger.error(f"Not all required charts for Gemini prompt were generated ({charts_generated_count}/{len(timeframes_for_gemini_prompt)}). Skipping Gemini analysis.")
            return

        # 4. Create composite image
        logger.info("Creating composite chart from individual timeframes...")
        composite_image_path = create_composite_chart(saved_chart_paths_for_gemini, CONTRACT_DISPLAY_SYMBOL)
        
        if not composite_image_path:
            logger.error("Failed to create composite chart. Skipping Gemini analysis.")
            return

        # 5. Analyze with Gemini using the composite image
        unified_gemini_analysis_text = None
        try:
            logger.info("Sending composite chart to Gemini for analysis...")
            unified_gemini_analysis_text = analyze_composite_chart_with_gemini(composite_image_path, CONTRACT_DISPLAY_SYMBOL)
        except Exception as e:
            logger.error(f"Error during Gemini analysis of composite chart: {e}", exc_info=True)
        
        # 6. Decide and Act on new trade opportunities
        if unified_gemini_analysis_text:
            gemini_trade_decision = parse_gemini_multi_chart_analysis(unified_gemini_analysis_text)
            make_final_trade_from_gemini_decision(gemini_trade_decision, all_data)
        else:
            logger.info("No Gemini analysis text received. No new trading decision taken.")
    
    logger.info(f"--- Trading Bot Cycle Content Ended: {datetime.now()} ---")

# --- Scheduling Functions ---
def scheduled_bot_task(sc):
    """Scheduled task that runs the trading bot cycle."""
    global shutdown_flag, first_run_after_start_time_done
    if shutdown_flag:
        logger.info("Shutdown initiated, not running scheduled task.")
        return

    logger.info("Scheduled bot task triggered.")
    try:
        trading_bot_cycle_content()
        first_run_after_start_time_done = True
    except Exception as e:
        logger.error(f"Unhandled exception in scheduled_bot_task: {e}", exc_info=True)
    finally:
        if not shutdown_flag:
            logger.info(f"Scheduling next run in {CYCLE_INTERVAL_MINUTES} minutes.")
            s.enter(CYCLE_INTERVAL_MINUTES * 60, 1, scheduled_bot_task, (sc,))

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_flag, oco_manager
    logger.info(f"Shutdown signal ({signal.Signals(sig).name}) received. Bot will stop after current tasks.")
    shutdown_flag = True
    
    # Stop OCO monitoring
    if oco_manager:
        oco_manager.stop_monitoring()
    
    # Cancel all pending scheduled events
    if hasattr(s, 'queue') and s.queue:
        logger.info(f"Cancelling {len(s.queue)} pending scheduled tasks.")
        for event in s.queue:
            try:
                s.cancel(event)
            except ValueError:
                pass
    logger.info("No new cycles will be scheduled.")

# --- OCO Management Functions ---
def get_oco_status_report():
    """Generate a status report of all OCO groups and global position limits."""
    global oco_manager
    
    if not oco_manager:
        return "OCO Manager not initialized"
    
    # Get current total position size
    current_total_contracts = get_current_total_position_size()
    
    active_groups = oco_manager.get_active_oco_groups()
    completed_trades = len(oco_manager.trade_records)
    
    report = f"\n=== OCO & Position Status Report ===\n"
    report += f"Total Position: {current_total_contracts}/{MAX_TOTAL_CONTRACTS} contracts\n"
    report += f"Available Capacity: {MAX_TOTAL_CONTRACTS - current_total_contracts} contracts\n"
    report += f"Active OCO Groups: {len(active_groups)}\n"
    report += f"Completed Trades: {completed_trades}\n\n"
    
    # Warning if approaching or exceeding limits
    if current_total_contracts >= MAX_TOTAL_CONTRACTS:
        report += "⚠️  WARNING: At or above maximum total position limit!\n\n"
    elif current_total_contracts >= MAX_TOTAL_CONTRACTS * 0.8:
        report += f"⚠️  CAUTION: Using {(current_total_contracts/MAX_TOTAL_CONTRACTS)*100:.1f}% of position capacity\n\n"
    
    for position_id, oco_group in active_groups.items():
        report += f"Position ID: {position_id}\n"
        report += f"  Contract: {oco_group['contract_id']}\n"
        report += f"  Side: {oco_group['trade_side']}\n"
        report += f"  Size: {oco_group.get('position_size', 0)} contracts\n"
        report += f"  Entry Price: {oco_group['entry_price']}\n"
        report += f"  Main Order Filled: {oco_group['main_order_filled']}\n"
        report += f"  Position Active: {oco_group['position_active']}\n"
        report += f"  Unrealized P&L: ${oco_group.get('unrealized_pnl', 0):.2f}\n"
        report += f"  Created: {oco_group['created_timestamp']}\n\n"
    
    return report

def force_close_all_oco_groups(reason="Manual shutdown"):
    """Force close all active OCO groups."""
    global oco_manager
    
    if not oco_manager:
        logger.warning("OCO Manager not initialized, cannot force close groups")
        return
    
    active_groups = list(oco_manager.get_active_oco_groups().keys())
    
    for position_id in active_groups:
        try:
            oco_manager.force_close_oco_group(position_id, reason)
            logger.info(f"Force closed OCO group {position_id}")
        except Exception as e:
            logger.error(f"Error force closing OCO group {position_id}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FuturesDesk Trading Bot with OCO Management...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize APIs
    if not initialize_futuresdesk_api():
        logger.critical("CRITICAL: Failed to initialize FuturesDesk API. Exiting.")
        exit(1)
    if not initialize_gemini_api():
        logger.critical("CRITICAL: Failed to initialize Gemini API. Exiting.")
        exit(1)
    if not initialize_oco_manager():
        logger.critical("CRITICAL: Failed to initialize OCO Manager. Exiting.")
        exit(1)

    logger.info(f"Bot configured for: {CONTRACT_DISPLAY_SYMBOL} (ID: {CONTRACT_ID}) on Account: {ACCOUNT_ID}")
    logger.info(f"Target start time: {TARGET_START_HOUR_EST:02d}:{TARGET_START_MINUTE_EST:02d} EST")
    logger.info(f"Cycle interval: {CYCLE_INTERVAL_MINUTES} minutes after initial start.")
    logger.info(f"Risk management: Max risk ${MAX_RISK_PER_TRADE} per trade, Max {MAX_CONTRACTS} contracts per trade, Max {MAX_TOTAL_CONTRACTS} total contracts")
    logger.info(f"OCO monitoring interval: {OCO_CHECK_INTERVAL_SECONDS} seconds")
    logger.info("To stop the bot, press Ctrl+C.")

    est_timezone = pytz.timezone('America/New_York')
    
    # Wait for target start time
    while not shutdown_flag:
        now_utc = datetime.now(pytz.utc)
        now_est = now_utc.astimezone(est_timezone)
        
        target_start_dt_est_today = now_est.replace(hour=TARGET_START_HOUR_EST, minute=TARGET_START_MINUTE_EST, second=0, microsecond=0)
        
        if now_est >= target_start_dt_est_today and not first_run_after_start_time_done:
            logger.info(f"Current EST time {now_est.strftime('%H:%M:%S')} is at or after target start time {target_start_dt_est_today.strftime('%H:%M:%S')}. Starting first cycle.")
            s.enter(0, 1, scheduled_bot_task, (s,))
            break
        elif now_est < target_start_dt_est_today:
            wait_seconds = (target_start_dt_est_today - now_est).total_seconds()
            logger.info(f"Waiting for target start time. Current EST: {now_est.strftime('%H:%M:%S')}. Target: {target_start_dt_est_today.strftime('%H:%M:%S')}. Sleeping for {wait_seconds:.0f} seconds.")
            # Sleep in chunks to allow for shutdown signal
            for _ in range(int(wait_seconds // 60) + 1):
                if shutdown_flag: break
                time.sleep(min(60, wait_seconds - (_ * 60)))
            if not shutdown_flag:
                s.enter(0, 1, scheduled_bot_task, (s,))
            break 
        else:
            # Calculate tomorrow's start time
            tomorrow_est = now_est + timedelta(days=1)
            target_start_dt_est_tomorrow = tomorrow_est.replace(hour=TARGET_START_HOUR_EST, minute=TARGET_START_MINUTE_EST, second=0, microsecond=0)
            wait_seconds = (target_start_dt_est_tomorrow - now_est).total_seconds()
            logger.info(f"Target start time {TARGET_START_HOUR_EST:02d}:{TARGET_START_MINUTE_EST:02d} EST for today has passed or first run completed. Waiting for tomorrow. Sleeping for {wait_seconds:.0f} seconds.")
            # Sleep in chunks
            for _ in range(int(wait_seconds // 60) + 1):
                if shutdown_flag: break
                time.sleep(min(60, wait_seconds - (_ * 60)))
            if not shutdown_flag:
                s.enter(0, 1, scheduled_bot_task, (s,))
            break

    # Run the scheduler
    if not shutdown_flag:
        try:
            logger.info("Starting scheduler...")
            s.run()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by KeyboardInterrupt (Ctrl+C).")
            shutdown_flag = True
        except Exception as e:
            logger.critical(f"Scheduler failed: {e}", exc_info=True)
    
    # Cleanup
    logger.info("Bot shutting down...")
    
    # Print final OCO status report
    logger.info(get_oco_status_report())
    
    # Force close any remaining OCO groups
    force_close_all_oco_groups("Bot shutdown")
    
    # Stop OCO monitoring thread
    if oco_manager:
        oco_manager.stop_monitoring()
    
    # Wait for monitoring thread to finish
    if oco_monitor_thread and oco_monitor_thread.is_alive():
        logger.info("Waiting for OCO monitoring thread to finish...")
        oco_monitor_thread.join(timeout=5)
    
    logger.info("FuturesDesk Trading Bot has terminated gracefully.")
