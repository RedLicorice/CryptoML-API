from cryptoml_core.models.trading import Asset, Position, Order, Equity, Baseline
from cryptoml_core.deps.mongodb.document_repository import DocumentRepository, DocumentNotFoundException, get_uuid
from pymongo import ASCENDING, DESCENDING
from typing import List, Optional


class AssetRepository(DocumentRepository):
    __collection__ = 'assets'
    __model__ = Asset

    def get_by_symbol(self, pipeline: str, dataset: str, target: str, symbol: str, window: int):
        query = {"pipeline": pipeline, "dataset": dataset, "target": target, "symbol": symbol, "window": window}
        document = self.collection.find_one(query)
        if not document:
            return None
        return self.__model__.parse_obj(document)

    def create_by_symbol(self, pipeline: str, dataset: str, target: str, symbol: str, window: int, fiat: float, balance: Optional[float] = 0.0):
        a = Asset(
            pipeline=pipeline,
            dataset=dataset,
            target=target,
            symbol=symbol,
            window=window,
            fiat=fiat,
            balance=balance
        )
        return self.create(a)

    def get_open_positions(self, asset_id: str):
        result = self.collection.aggregate([
            {
                '$match': {'id': asset_id}
            },
            {
                '$project': {
                    'positions': {
                        '$filter': {
                            'input': '$positions',
                            'as': 'position',
                            'cond': {'$eq': ['$$position.status', 'OPEN']}
                        }
                    }
                }
            }
        ])
        return [Position.parse_obj(p) for p in [d['positions'] for d in result][0]]
