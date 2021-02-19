from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm.session import Session
from cryptoml_core.deps.config import config


engine = None
Base = declarative_base()
# Create missing tables
db_session = None

def init_engine(**kwargs):
  global engine, db_session
  uri = config['database']['sql']['url'].get(str)
  if uri.startswith('sqlite://'):
      kwargs.update({'connect_args': {'check_same_thread': False}})
  engine = create_engine(uri, **kwargs)
  db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
  return engine

def init_db():
    if not engine:
        return
    Base.metadata.create_all(bind=engine)

def get_session() -> Session:
    global db_session
    if not db_session:
        # raise Exception("Database engine not initialized!")
        init_engine()
        init_db()
    return db_session()

