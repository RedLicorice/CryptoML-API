class BaseRepository:
    def __init__(self, sessionFactory):
        self.session = sessionFactory()

    def __del__(self):
        self.session.close()

    def commit(self):
        self.session.commit()

    def add(self, entity, commit=False):
        self.session.add(entity)
        if commit:
            self.commit()