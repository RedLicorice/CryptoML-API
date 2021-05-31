from cryptoml_core.exceptions import MessageException
from cryptoml_core.repositories.trading_repository import AssetRepository, Order, Position, Asset, Equity, get_uuid
import pandas as pd
from cryptoml_core.util.timestamp import timestamp_diff

MARGIN_SHORT_FIXED_FEE = 0.0001  # 0.01% fee at position open
MARGIN_SHORT_DAILY_FEE = 0.0002 * 5  # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
MARGIN_LONG_FIXED_FEE = 0.0001  # 0.01% fee at position open
MARGIN_LONG_DAILY_FEE = 0.0001 * 5  # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
SPOT_FIXED_FEE = 0.0026  # 0.16-0.26% fee at every spot transaction (maker-taker)
FIAT_LOAN_LIMIT = 5000
COLL_LOAN_LIMIT = 0.5

class TradingService:
    def __init__(self):
        self.repo = AssetRepository()

    def get_asset(self, pipeline: str, dataset: str, target: str, symbol: str, window: int, create=True):
        res = self.repo.get_by_symbol(pipeline, dataset, target, symbol, window=window)
        if not res:
            if not create:
                return None
            res = self.repo.create_by_symbol(pipeline, dataset, target, symbol, window, fiat=10000, balance=0.0)
        return res

    @staticmethod
    def get_equity(asset: Asset, price: float):
        return round((asset.fiat - asset.fiat_loan) + (asset.balance - asset.coll_loan) * price, 3)

    def update_equity(self, asset: Asset, day: str, price: float):
        e = Equity(
            timestamp=day,
            equity=TradingService.get_equity(asset=asset, price=price),
            num_long=asset.num_long,
            num_short=asset.num_short,
            num_spot=asset.num_spot
        )
        # Append open order
        asset.equities.append(e)

        # Update the asset instance in DB and return the result
        return self.repo.update(asset.id, asset)

    @staticmethod
    def parse_equity_df(asset: Asset):
        # Re-convert asset's equity history to a DataFrame
        results = pd.DataFrame([e.dict() for e in asset.equities])
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        results.index = pd.to_datetime(results.timestamp)

        return results.drop(labels='timestamp', axis=1)

    @staticmethod
    def parse_orders_df(asset: Asset):
        # Re-convert asset's order history to a DataFrame
        results = pd.DataFrame([o.dict() for o in asset.orders])
        # Parse index so it's a DateTimeIndex, because Mongo stores it as a string
        results.index = pd.to_datetime(results.timestamp)
        return results.drop(labels='timestamp', axis=1)

    @staticmethod
    def get_position_size_amount(asset: Asset, close_price: float, size: float):
        # We need to determine operation price (op_price), and we can do this in two ways:
        # - Fixed fractional (we invest a fixed fraction of our total equity value)
        # - Fixed amount (we invest a fixed FIAT amount per trade)
        # Since it would make no sense to trade only 1$, we distinguish the two cases by the "size" parameter:
        # If size < 1 consider it a percentage of total equity for this asset (Fixed fractional)
        if size <= 1:
            op_price = TradingService.get_equity(asset, close_price) * size
        # Else consider "size" the total worth of our position
        #  EXAMPLE: suppose close price for pair BTCUSD is $50000 we want to open a position with size=100$,
        #  amount will be:
        #      amount = 100$/50000$ = 0.002 BTC
        else:
            op_price = size
        amount = op_price / close_price
        return round(op_price, 3), round(amount, 8)

    @staticmethod
    def apply_price_change(price, pct):
        # When calculating stop loss, "pct" should be negative for LONG trades, positive for SHORT trades.
        # The opposite is true for take profit!
        return round(price + (price * pct), 3)

    @staticmethod
    def get_percent_change(new_price, old_price):
        if not old_price:
            raise ValueError("Divide by Zero: Old price is none!")
        return round((new_price - old_price) / old_price, 5)

    def open_long(self, asset: Asset, day: str, close_price: float, size: float, **kwargs):
        # Get operation's fiat value and collateral amount involved in this position
        op_price, amount = TradingService.get_position_size_amount(asset, close_price, size)

        # Margin long trades incur an opening fixed fee in FIAT, paid immediately
        op_fee = round(op_price * MARGIN_LONG_FIXED_FEE, 3)

        # Fiat wallet must hold the position's price (in order to be able to pay back the loan in the future)
        #  as well as the fees, which are paid immediately
        need_fiat = op_price + op_fee
        if asset.fiat < need_fiat:
            raise MessageException("Not enough fiat to open LONG position on "
                                   f"{asset.symbol}: Wallet: {asset.fiat} Needed: {need_fiat}")
        new_loan = round(asset.fiat_loan + op_price, 3)
        if new_loan > FIAT_LOAN_LIMIT:
            raise MessageException("Not enough allowance to open LONG position on "
                                   f"{asset.symbol}: Cur_loan: {asset.fiat_loan} "
                                   f"New loan: {new_loan} "
                                   f"Need allowance: {new_loan - FIAT_LOAN_LIMIT}")
        # Create the position instance
        p = Position(
            id=get_uuid(),
            type='MARGIN_LONG',
            status='OPEN',
            open_price=close_price,  # Position is opened at market close price (approximation)
            open_timestamp=day,  # Day OF THE SIMULATION!
            amount=amount,  # Collateral amount
            coll_loan=0.0,  # This is a MARGIN LONG trade so we don't borrow collateral
            fiat_loan=op_price,  # This is a MARGIN LONG trade so we borrow FIAT
            # Set default stop loss to 3% below opening price
            stop_loss=TradingService.apply_price_change(close_price, kwargs.get('stop_loss')),
            # Set a take profit at 5% gains
            take_profit=TradingService.apply_price_change(close_price, kwargs.get('take_profit')),
            last_price=close_price,  # Reference price for stop loss
            open_fee=op_fee
        )
        # Create the order instance
        o = Order(
            id=get_uuid(),
            position_id=p.id,
            timestamp=day,
            type='OPEN_LONG',
            amount=amount,
            price=close_price,
            detail=kwargs.get('detail'),
            change=None
        )

        # Perform updates to the asset
        # Opening fee is paid immediately from FIAT wallet
        asset.fiat = round(asset.fiat - op_fee, 3)
        # Update fiat_loan counter with the amount we borrowed for this position
        asset.fiat_loan = round(asset.fiat_loan + op_price, 3)
        # Update collateral wallet with the amount we purchased
        asset.balance = round(asset.balance + amount, 8)
        # Update open long counter
        asset.num_long += 1
        # Append the position
        asset.positions.append(p)
        # Append open order
        asset.orders.append(o)

        # Update the asset instance in DB and return the result
        return self.repo.update(asset.id, asset)

    @staticmethod
    def get_position_index(asset: Asset, position: Position):
        for i in range(len(asset.positions)):
            if asset.positions[i].id == position.id:
                return i
        return None

    def close_long(self, asset: Asset, day: str, close_price: float, position: Position, **kwargs):
        # Get the index of this position inside the asset's positions so we can replace it with the updated one
        index = TradingService.get_position_index(asset, position)
        if index is None:
            raise MessageException(
                f"LONG Position {position.id} is not related to asset {asset.symbol} with ID {asset.id}!")

        if asset.balance < position.amount:
            raise MessageException("Not enough collateral to close LONG position on "
                                   f"{asset.symbol}: Wallet: {asset.balance} Needed: {position.amount}")

        # In order to calculate the rolling fee, we need to apply the fees to the loan amount
        # then multiply it by the number of days the loan was active
        num_days = timestamp_diff(day, position.open_timestamp) / 86400  # Number of days this position was open
        sell_fee = round(position.fiat_loan * MARGIN_LONG_DAILY_FEE * num_days, 3)

        # Calculate the revenue from selling the tokens
        sell_revenue = round(position.amount * close_price, 3)
        # Pay back loan + interest to obtain booked profit for this position
        booked_profit = round(sell_revenue - position.fiat_loan - sell_fee, 3)

        # Update the position with closing details
        position.status = 'CLOSED'
        position.close_price = close_price
        position.close_timestamp = day
        position.close_fee = sell_fee
        position.price_change = TradingService.get_percent_change(position.close_price, position.open_price)
        position.profit = booked_profit

        # Create the order instance
        o = Order(
            id=get_uuid(),
            position_id=position.id,
            timestamp=day,
            type='CLOSE_LONG',
            amount=position.amount,
            price=close_price,
            detail=kwargs.get('detail'),
            change=TradingService.get_percent_change(position.close_price, position.open_price)
        )

        # Perform updates to the asset
        asset.fiat = round(asset.fiat + booked_profit, 3)
        # Update fiat_loan counter with the amount we borrowed for this position
        asset.fiat_loan = round(asset.fiat_loan - position.fiat_loan, 3)
        # Remove the collateral we sold from balance
        asset.balance = round(asset.balance - position.amount, 8)
        # Update open long counter
        asset.num_long -= 1
        # Update the position by index
        asset.positions[index] = position
        # Append close order
        asset.orders.append(o)

        return self.repo.update(asset.id, asset)

    def open_short(self, asset: Asset, day: str, close_price: float, size: float, **kwargs):
        # Get operation's fiat value and collateral amount involved in this position
        op_value, amount = TradingService.get_position_size_amount(asset, close_price, size)

        # Margin short trades incur an opening fixed fee, which is paid immediately
        #  Since SHORT fees are paid in collateral, we need to buy some in SPOT, and pay fees for it as well!
        op_fee = amount * MARGIN_SHORT_FIXED_FEE
        op_fee = round(op_fee + op_fee * SPOT_FIXED_FEE, 8)
        # Convert final opening fee in FIAT
        op_fee = round(op_fee * close_price, 3)

        # for short orders, FIAT wallet should hold 1.5x the position price as per most of the exchanges' rules
        #  but we relapse this and approximate to 1.0x plus opening fees
        need_fiat = op_value + op_fee
        if asset.fiat < need_fiat:
            raise MessageException("Not enough fiat to open SHORT position on "
                                   f"{asset.symbol}: Wallet: {asset.fiat} Needed: {need_fiat}")
        new_loan = round(asset.coll_loan + amount, 8)
        if new_loan > COLL_LOAN_LIMIT:
            raise MessageException("Not enough allowance to open SHORT position on "
                                   f"{asset.symbol}: Cur_loan: {asset.coll_loan} "
                                   f"New loan: {new_loan} "
                                   f"Need allowance: {new_loan - COLL_LOAN_LIMIT}")
        # In short orders, we immediately sell borrowed collateral for fiat at market price
        # Since fees are paid immediately, they are paid from this sale's revenue
        sell_revenue = op_value - op_fee

        # Create the position instance
        p = Position(
            id=get_uuid(),
            type='MARGIN_SHORT',
            status='OPEN',
            open_price=close_price,  # Position is opened at market close price (approximation)
            open_timestamp=day,  # Day OF THE SIMULATION!
            amount=amount,  # Collateral amount
            coll_loan=amount,  # This is a MARGIN SHORT trade so we borrow collateral
            fiat_loan=0.0,  # This is a MARGIN LONG trade so we don't borrow FIAT
            # Set default stop loss to 5% above opening price
            stop_loss=TradingService.apply_price_change(close_price, kwargs.get('stop_loss')),
            # Set a take profit at 5% profit
            take_profit=TradingService.apply_price_change(close_price, kwargs.get('take_profit')),
            last_price=close_price,  # Reference price for stop loss
            open_fee=op_fee
        )
        # Create the order instance
        o = Order(
            id=get_uuid(),
            position_id=p.id,
            timestamp=day,
            type='OPEN_SHORT',
            amount=amount,
            price=close_price,
            detail=kwargs.get('detail'),
            change=None
        )

        # Perform updates to the asset
        # Opening fee is paid immediately from FIAT wallet
        asset.fiat = round(asset.fiat + sell_revenue, 3)
        # Update coll_loan counter with the amount we borrowed for this position
        asset.coll_loan = round(asset.coll_loan + amount, 8)

        # Update open short counter
        asset.num_short += 1
        # Append the position
        asset.positions.append(p)
        # Append open order
        asset.orders.append(o)

        # Update the asset instance in DB and return the result
        return self.repo.update(asset.id, asset)

    def close_short(self, asset: Asset, day: str, close_price: float, position: Position, **kwargs):
        # Get the index of this position inside the asset's positions so we can replace it with the updated one
        index = TradingService.get_position_index(asset, position)
        if index is None:
            raise MessageException(
                f"SHORT Position {position.id} is not related to asset {asset.symbol} with ID {asset.id}!")

        # In order to calculate the rolling fee, we need to apply the fees to the loan amount
        # then multiply it by the number of days the loan was active
        num_days = timestamp_diff(day, position.open_timestamp) / 86400  # Number of days this position was open
        sell_fee = round(position.coll_loan * MARGIN_LONG_DAILY_FEE * num_days, 8)
        # Since we buy back our loan at spot market, we need to add SPOT fee
        spot_fee = round((position.coll_loan + sell_fee) * SPOT_FIXED_FEE, 8)
        # Sell fee is in collateral
        sell_fee = sell_fee + spot_fee

        # Total buyback price is sell fee (which is loan interest + spot fee) + loan amount
        buyback_price = round((sell_fee + position.coll_loan) * close_price, 3)

        if asset.fiat < buyback_price:
            raise MessageException("Not enough fiat to close SHORT position on "
                                   f"{asset.symbol}: Wallet: {asset.fiat} Needed: {buyback_price}")

        # Position profit is opening price minus buyback price
        open_revenue = round(position.open_price * position.amount, 3) - position.open_fee
        booked_profit = open_revenue - buyback_price

        # Update the position with closing details
        position.status = 'CLOSED'
        position.close_price = close_price
        position.close_timestamp = day
        position.close_fee = sell_fee
        position.price_change = TradingService.get_percent_change(position.close_price, position.open_price)
        position.profit = booked_profit

        # Create the order instance
        o = Order(
            id=get_uuid(),
            position_id=position.id,
            timestamp=day,
            type='CLOSE_SHORT',
            amount=position.amount,
            price=close_price,
            detail=kwargs.get('detail'),
            change=TradingService.get_percent_change(position.close_price, position.open_price)
        )

        # Perform updates to the asset
        # Opening fee is paid immediately from FIAT wallet
        asset.fiat = round(asset.fiat - buyback_price, 3)
        # Update fiat_loan counter with the amount we borrowed for this position
        asset.coll_loan = round(asset.coll_loan - position.coll_loan, 8)

        # Update open long counter
        asset.num_short -= 1
        # Update the position by index
        asset.positions[index] = position
        # Append close order
        asset.orders.append(o)

        return self.repo.update(asset.id, asset)

    def get_open_positions(self, asset: Asset, day: str):
        positions = self.repo.get_open_positions(asset.id)
        return positions

    @staticmethod
    def get_position_age(position: Position, day: str):
        return timestamp_diff(day, position.open_timestamp)

    @staticmethod
    def check_stop_loss(position: Position, low_price: float):
        if position.type == 'MARGIN_LONG':
            return low_price < position.stop_loss
        elif position.type == 'MARGIN_SHORT':
            return low_price > position.stop_loss

    def update_stop_loss(self, asset: Asset, position: Position, close_price: float, pct: float):
        # Get the index of this position inside the asset's positions so we can replace it with the updated one
        index = TradingService.get_position_index(asset, position)
        if index is None:
            raise MessageException(
                f"[SL Update] Position {position.id} is not related to asset {asset.symbol} with ID {asset.id}!")
        position.stop_loss = TradingService.apply_price_change(close_price, pct)
        asset.positions[index] = position
        return self.repo.update(asset.id, asset)

    def update_take_profit(self, asset: Asset, position: Position, close_price: float, pct: float):
        # Get the index of this position inside the asset's positions so we can replace it with the updated one
        index = TradingService.get_position_index(asset, position)
        if index is None:
            raise MessageException(
                f"[TP Update] Position {position.id} is not related to asset {asset.symbol} with ID {asset.id}!")
        position.take_profit = TradingService.apply_price_change(close_price, pct)
        asset.positions[index] = position
        return self.repo.update(asset.id, asset)

    @staticmethod
    def check_take_profit(position: Position, high_price: float):
        if position.type == 'MARGIN_LONG':
            return high_price >= position.take_profit
        elif position.type == 'MARGIN_SHORT':
            return high_price <= position.take_profit
