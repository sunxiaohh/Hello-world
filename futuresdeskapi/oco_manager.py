import threading
import time
import logging # Optional: for logging within the manager

# Configure basic logging for the OCO Manager (optional)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - OCOManager - %(levelname)s - %(message)s')

class OcoManager:
    def __init__(self, order_api_client, position_api_client, polling_interval=10):
        self.order_api_client = order_api_client
        self.position_api_client = position_api_client
        self.active_oco_orders = {}  # Stores active OCO pairs
        self.polling_interval = polling_interval  # seconds
        self._lock = threading.Lock() # To ensure thread-safe access to active_oco_orders

        self.polling_thread = None
        self.stop_event = threading.Event()

    def add_oco_pair(self, client_oco_id: str, sl_order_id: str, tp_order_id: str, account_id: int, contract_id: str):
        with self._lock:
            if client_oco_id in self.active_oco_orders:
                # logging.warning(f"OCO pair with ID {client_oco_id} already exists. Overwriting.")
                pass # Or raise error
            self.active_oco_orders[client_oco_id] = {
                "sl_order_id": sl_order_id,
                "tp_order_id": tp_order_id,
                "account_id": account_id,
                "contract_id": contract_id,
                "status": "active" # Potential statuses: active, cancelling_sl, cancelling_tp, completed
            }
        # logging.info(f"Added OCO pair {client_oco_id}: SL {sl_order_id}, TP {tp_order_id} for contract {contract_id}")

    def remove_oco_pair(self, client_oco_id: str):
        with self._lock:
            if client_oco_id in self.active_oco_orders:
                del self.active_oco_orders[client_oco_id]
                # logging.info(f"Removed OCO pair {client_oco_id}")
                return True
            # logging.warning(f"Attempted to remove non-existent OCO pair {client_oco_id}")
            return False

    def _get_active_oco_pairs_copy(self):
        with self._lock:
            return dict(self.active_oco_orders) # Return a shallow copy

    def _poll_oco_orders(self):
        # logging.info("OCO polling thread started.")
        while not self.stop_event.is_set():
            try:
                active_pairs = self._get_active_oco_pairs_copy()
                if not active_pairs:
                    self.stop_event.wait(self.polling_interval) # Wait if no active OCOs
                    continue

                for client_oco_id, oco_data in active_pairs.items():
                    if self.stop_event.is_set(): break # Exit early if stop requested

                    # logging.debug(f"Polling OCO pair: {client_oco_id}")

                    # 1. Check position status first
                    try:
                        open_positions = self.position_api_client.search_open_positions(account_id=oco_data["account_id"])
                        position_exists_for_contract = any(
                            p['contractId'] == oco_data["contract_id"] and abs(p.get('size', 0)) > 0 # Assuming 'size' indicates position quantity
                            for p in open_positions
                        )

                        if not position_exists_for_contract:
                            # logging.info(f"Position for contract {oco_data['contract_id']} (OCO: {client_oco_id}) is closed or zero. Cancelling both legs.")
                            self._cancel_leg(oco_data["account_id"], oco_data["sl_order_id"], "SL", client_oco_id)
                            self._cancel_leg(oco_data["account_id"], oco_data["tp_order_id"], "TP", client_oco_id)
                            self.remove_oco_pair(client_oco_id)
                            continue 
                    except Exception as e:
                        # logging.error(f"Error checking position status for OCO {client_oco_id}: {e}")
                        # Decide if we should continue or skip this pair for now
                        continue # For now, skip to next OCO pair on error

                    # 2. Fetch open orders for the account
                    try:
                        open_api_orders = self.order_api_client.search_open_orders(account_id=oco_data["account_id"])
                        open_order_ids = {str(o['id']) for o in open_api_orders} # API might return int or str for orderId
                    except Exception as e:
                        # logging.error(f"Error fetching open orders for account {oco_data['account_id']} (OCO: {client_oco_id}): {e}")
                        continue # Skip this OCO pair for this cycle

                    sl_id_str = str(oco_data["sl_order_id"])
                    tp_id_str = str(oco_data["tp_order_id"])

                    # 3. Check SL order
                    if sl_id_str not in open_order_ids:
                        # logging.info(f"SL order {sl_id_str} (OCO: {client_oco_id}) not open (filled/cancelled). Cancelling TP order {tp_id_str}.")
                        self._cancel_leg(oco_data["account_id"], tp_id_str, "TP", client_oco_id)
                        self.remove_oco_pair(client_oco_id)
                        continue

                    # 4. Check TP order
                    if tp_id_str not in open_order_ids:
                        # logging.info(f"TP order {tp_id_str} (OCO: {client_oco_id}) not open (filled/cancelled). Cancelling SL order {sl_id_str}.")
                        self._cancel_leg(oco_data["account_id"], sl_id_str, "SL", client_oco_id)
                        self.remove_oco_pair(client_oco_id)
                        continue
                    
                    # logging.debug(f"OCO pair {client_oco_id} is still active.")

            except Exception as e:
                # logging.error(f"Error in OCO polling loop: {e}")
                # Avoid crashing the thread. Specific errors should be handled above.
                pass
            
            self.stop_event.wait(self.polling_interval) # Wait before next poll cycle
        # logging.info("OCO polling thread stopped.")

    def _cancel_leg(self, account_id, order_id_to_cancel, leg_name, client_oco_id_context):
        try:
            # First, check if it's still open to avoid redundant cancellations if API is slow
            # This check might be too much if search_open_orders is slow.
            # open_orders_check = self.order_api_client.search_open_orders(account_id=account_id)
            # if not any(str(o['id']) == str(order_id_to_cancel) for o in open_orders_check):
            #     logging.info(f"{leg_name} order {order_id_to_cancel} (OCO: {client_oco_id_context}) already not open. No cancellation needed.")
            #     return

            self.order_api_client.cancel_order(account_id=account_id, order_id=order_id_to_cancel)
            # logging.info(f"Successfully cancelled {leg_name} order {order_id_to_cancel} (OCO: {client_oco_id_context}).")
        except Exception as e:
            # logging.error(f"Failed to cancel {leg_name} order {order_id_to_cancel} (OCO: {client_oco_id_context}): {e}")
            # Further error handling/retry could be added here.
            # If cancellation fails, the order might remain active.
            pass


    def start(self):
        if self.polling_thread is not None and self.polling_thread.is_alive():
            # logging.warning("OCO polling thread already running.")
            return

        self.stop_event.clear()
        self.polling_thread = threading.Thread(target=self._poll_oco_orders, daemon=True)
        self.polling_thread.start()
        # logging.info("OCO Manager started.")

    def stop(self):
        # logging.info("Stopping OCO Manager...")
        self.stop_event.set()
        if self.polling_thread is not None and self.polling_thread.is_alive():
            self.polling_thread.join()
        # logging.info("OCO Manager stopped.")
