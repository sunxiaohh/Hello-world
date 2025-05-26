from .auth import AuthClient
from .account import AccountAPI
from .contract import ContractAPI
from .order import OrderAPI
from .position import PositionAPI
from .trade import TradeAPI
from .history import HistoryAPI
from .realtime import RealTimeClient
from .oco_manager import OcoManager
# Optional: import logging if uncommenting logging lines
# import logging


class FuturesDeskClient:
    def __init__(self, username: str, api_key: str, base_url: str = "https://api.thefuturesdesk.projectx.com"):
        self.auth = AuthClient(base_url)
        self.auth.login(username, api_key)
        self.token = self.auth.get_token()
        self.base_url = base_url

        self.account = AccountAPI(self.token, self.base_url)
        self.contract = ContractAPI(self.token, self.base_url)
        self.order = OrderAPI(self.token, self.base_url)
        self.position = PositionAPI(self.token, self.base_url)
        self.trade = TradeAPI(self.token, self.base_url)
        self.history = HistoryAPI(self.token, self.base_url)
        self.oco_manager = OcoManager(order_api_client=self.order, position_api_client=self.position)

    def place_oco_order(
        self,
        account_id: int,
        contract_id: str,
        position_side: int,
        size: int,
        stop_loss_price: float,
        take_profit_price: float,
        custom_tag_sl: str = None,
        custom_tag_tp: str = None
    ):
        '''
        Places an OCO (One-Cancels-the-Other) order and registers it with the OCO manager.
        Returns:
            tuple: (sl_order_id, tp_order_id, client_oco_id) if successful.
        Raises:
            Exception from self.order.place_oco_order if placement fails.
        '''
        # The place_oco_order in OrderAPI will raise an exception if a leg fails,
        # potentially including the ID of a successfully placed first leg.
        sl_order_id, tp_order_id = self.order.place_oco_order(
            account_id=account_id,
            contract_id=contract_id,
            position_side=position_side,
            size=size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            custom_tag_sl=custom_tag_sl,
            custom_tag_tp=custom_tag_tp
        )

        # If the above call succeeded, both sl_order_id and tp_order_id are valid.
        client_oco_id = f"oco_{sl_order_id}_{tp_order_id}"
        self.oco_manager.add_oco_pair(
            client_oco_id=client_oco_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
            account_id=account_id,
            contract_id=contract_id
        )
        # logging.info(f"FuturesDeskClient: Registered OCO pair {client_oco_id} for contract {contract_id}")
        return sl_order_id, tp_order_id, client_oco_id

    def start_oco_manager(self, polling_interval: int = None):
        '''Starts the OCO manager polling thread.'''
        if hasattr(self, 'oco_manager') and self.oco_manager:
            if polling_interval is not None:
                self.oco_manager.polling_interval = polling_interval
            self.oco_manager.start()
            # logging.info("FuturesDeskClient: OCO Manager started.")
        else:
            # logging.error("FuturesDeskClient: OcoManager not initialized.")
            raise Exception("OcoManager not initialized.")

    def stop_oco_manager(self):
        '''Stops the OCO manager polling thread.'''
        if hasattr(self, 'oco_manager') and self.oco_manager:
            self.oco_manager.stop()
            # logging.info("FuturesDeskClient: OCO Manager stopped.")
        else:
            # logging.warning("FuturesDeskClient: OcoManager not initialized or already stopped.")
            pass # Or raise error if strictness is needed

    def __del__(self):
        # Attempt to stop the OCO manager when the client object is garbage collected.
        # Note: __del__ is not guaranteed to be called reliably in all Python implementations
        # or all circumstances (e.g., if objects are part of cycles).
        # Explicitly calling stop_oco_manager() is safer.
        if hasattr(self, 'oco_manager') and self.oco_manager:
            if self.oco_manager.polling_thread is not None and self.oco_manager.polling_thread.is_alive():
                # logging.info("FuturesDeskClient __del__: Stopping OCO manager.")
                self.oco_manager.stop()