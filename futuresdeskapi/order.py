import requests

class OrderAPI:
    def __init__(self, token: str, base_url: str):
        self.token = token
        self.base_url = base_url

    def search_orders(self, account_id: int, start_timestamp: str, end_timestamp: str = None):
        url = f"{self.base_url}/api/Order/search"
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "accountId": account_id,
            "startTimestamp": start_timestamp
        }
        if end_timestamp:
            payload["endTimestamp"] = end_timestamp

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("errorCode") == 0:
                return data.get("orders", [])
            else:
                raise Exception(f"API Error: {data.get('errorMessage')}")
        else:
            raise Exception(f"HTTP Error: {response.status_code} {response.text}")

    def place_oco_order(
        self,
        account_id: int,
        contract_id: str,
        position_side: int,  # Side of the main position (1 for Buy, 2 for Sell)
        size: int,
        stop_loss_price: float,
        take_profit_price: float,
        custom_tag_sl: str = None, # Optional custom tag for SL
        custom_tag_tp: str = None  # Optional custom tag for TP
    ):
        """
        Places a One-Cancels-the-Other (OCO) order, consisting of a stop-loss and a take-profit order.
        Note: The API does not natively support OCO orders. This function places two separate orders.
        It's the responsibility of a higher-level component or the user to manage these orders
        as an OCO group (e.g., cancel one if the other fills).
        """

        if position_side == 1:  # Main position is Buy
            sl_side = 2  # Sell
            tp_side = 2  # Sell
        elif position_side == 2:  # Main position is Sell
            sl_side = 1  # Buy
            tp_side = 1  # Buy
        else:
            raise ValueError("position_side must be 1 (Buy) or 2 (Sell)")

        # Assumptions for order types based on typical API behavior:
        # Type 3: Stop Order (for stop-loss)
        # Type 2: Limit Order (for take-profit)
        # These should be verified with the actual API documentation if available.
        stop_loss_order_type = 3
        take_profit_order_type = 2

        sl_order_id = None
        try:
            sl_order_id = self.place_order(
                account_id=account_id,
                contract_id=contract_id,
                type=stop_loss_order_type,
                side=sl_side,
                size=size,
                stop_price=stop_loss_price,
                custom_tag=custom_tag_sl
            )

            tp_order_id = self.place_order(
                account_id=account_id,
                contract_id=contract_id,
                type=take_profit_order_type,
                side=tp_side,
                size=size,
                limit_price=take_profit_price,
                custom_tag=custom_tag_tp
            )
            return sl_order_id, tp_order_id
        except Exception as e:
            # If the TP order fails after SL order was placed, include SL order ID in the exception.
            if sl_order_id is not None:
                raise Exception(f"Failed to place take-profit order. Stop-loss order (ID: {sl_order_id}) was placed. Original error: {e}")
            else:
                # Exception from placing SL order
                raise e

    def search_open_orders(self, account_id: int):
        url = f"{self.base_url}/api/Order/searchOpen"
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "accountId": account_id
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("errorCode") == 0:
                return data.get("orders", [])
            else:
                raise Exception(f"API Error: {data.get('errorMessage')}")
        else:
            raise Exception(f"HTTP Error: {response.status_code} {response.text}")

    def place_order(
        self,
        account_id: int,
        contract_id: str,
        type: int,
        side: int,
        size: int,
        limit_price=None,
        stop_price=None,
        trail_price=None,
        custom_tag=None,
        linked_order_id=None
    ):
        url = f"{self.base_url}/api/Order/place"
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": type,
            "side": side,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
            "customTag": custom_tag,
            "linkedOrderId": linked_order_id
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("errorCode") == 0:
                return data.get("orderId")
            else:
                raise Exception(f"API Error: {data.get('errorMessage')}")
        else:
            raise Exception(f"HTTP Error: {response.status_code} {response.text}")

    def cancel_order(self, account_id: int, order_id: int):
        url = f"{self.base_url}/api/Order/cancel"
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "accountId": account_id,
            "orderId": order_id
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("errorCode") == 0:
                return True
            else:
                raise Exception(f"API Error: {data.get('errorMessage')}")
        else:
            raise Exception(f"HTTP Error: {response.status_code} {response.text}")

    def modify_order(
        self,
        account_id: int,
        order_id: int,
        size=None,
        limit_price=None,
        stop_price=None,
        trail_price=None
    ):
        url = f"{self.base_url}/api/Order/modify"
        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        payload = {
            "accountId": account_id,
            "orderId": order_id,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("errorCode") == 0:
                return True
            else:
                raise Exception(f"API Error: {data.get('errorMessage')}")
        else:
            raise Exception(f"HTTP Error: {response.status_code} {response.text}")