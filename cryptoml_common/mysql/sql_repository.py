from cryptoml_common.mysql import Session, get_session


class SQLRepository:
    def __init__(self):
        self.session: Session = get_session()

    def commit(self):
        self.session.commit()

    def add(self, entity, commit=False):
        if isinstance(entity, list):
            for x in entity:
                self.session.add(x)
        else:
            self.session.add(entity)

        if commit:
            self.commit()