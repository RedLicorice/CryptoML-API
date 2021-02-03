from ..database import Session

class FeaturesRepository:
    def __init__(self, DBSession: Session):
        self.session = DBSession

    def persist(self, entity):
        self.session.add(entity)
        self.session.flush()
        return entity