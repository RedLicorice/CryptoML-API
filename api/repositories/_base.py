from ..database import Session, get_session

class BaseRepository:
    def __init__(self):
        self.session: Session = get_session()

    def commit(self):
        self.session.commit()

    def add(self, entity, commit=False):
        self.session.add(entity)
        if commit:
            self.commit()