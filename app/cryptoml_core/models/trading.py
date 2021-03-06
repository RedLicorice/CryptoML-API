from cryptoml_core.deps.mongodb.document_model import DocumentModel, BaseModel
from typing import Optional, List


class Order(BaseModel):
    id: Optional[str] = ''  # Unique identifier for the order subdocument
    position_id: str  # Position this order refers to
    timestamp: str  # Timestamp for operation
    type: str  # Type of operation OPEN_LONG/CLOSE_LONG/OPEN_SHORT/CLOSE_SHORT
    amount: float  # Collateral amount involved in the operation
    price: float  # Collateral exchange price at time of operation

    detail: Optional[str] = None
    change: Optional[float] = None  # Price change wrt. open (only for CLOSE)


class Position(BaseModel):
    id: Optional[str] = ''  # Unique identifier for the position subdocument
    type: str  # One of "SPOT", "MARGIN_SHORT", "MARGIN_LONG"
    status: str   # One of "OPEN", "CLOSED"

    open_price: float  # COLLATERAL price when position is opened (eg for BTCUSD this is USD)
    open_timestamp: str  # Position open timestamp

    close_price: Optional[float] = None  # COLLATERAL price when position is closed (eg for BTCUSD this is USD)
    close_timestamp: Optional[str] = None  # Position close timestamp

    amount: float  # Total size of the position
    coll_loan: Optional[float] = 0.0  # For margin short trades, collateral is borrowed from the exchange's liquidity pool and must be returned
    fiat_loan: Optional[float] = 0.0  # For margin long trades, fiat is borrowed from the exchange's liquidity pool and must be returned

    # Mitigate losses by immediately closing the position if price goes below this limit (or above in case of SHORT)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    last_price: Optional[float] = None  # Price when stop loss was last checked

    open_fee: Optional[float] = None  # Fee when position was opened (Fixed fee for margin trades)
    close_fee: Optional[float] = None  # Fee when position was closed (Rolling fee for margin trades)

    price_change: Optional[float] = None
    profit: Optional[float] = None

class Equity(BaseModel):
    timestamp: str
    equity: float
    num_long: int
    num_short: int
    num_spot: int

class Baseline(Equity):
    name: str

class TradingAccount(DocumentModel):
    fiat: float  # Amount of FIAT allocated for this asset and available for trades
    balance: float  # Amount of assets owned for this asset
    coll_loan: Optional[float] = 0.0  # Amount of this asset that should be returned to liquidity pool
    fiat_loan: Optional[float] = 0.0  # Amount of this asset that should be returned to liquidity pool
    num_long: Optional[int] = 0  # Number of LONG positions open for this asset
    num_short: Optional[int] = 0  # Number of SHORT positions open for this asset
    num_spot: Optional[int] = 0  # Number of SPOT positions open for this asset
    positions: Optional[List[Position]] = []  # A list containing all positions for this asset (eg. open price ..)
    orders: Optional[List[Order]] = []  # A list containing the order history for this asset (Eg. open position)
    equities: Optional[List[Equity]] = []  # A list containing equity history for this asset
    baselines: Optional[List[Baseline]] = []  # A list containing equity history for this asset


class Asset(TradingAccount):
    symbol: str  # Treating each asset independently, this is the pair name Eg. "BTCUSD"
    pipeline: str
    dataset: str
    target: str
    window: int

