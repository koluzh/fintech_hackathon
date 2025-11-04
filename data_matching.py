import pandas as pd
from sqlalchemy.orm import declarative_base
from sqlalchemy import String, Integer, Column, MetaData, Table, create_engine # noqa
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

df = pd.read_csv("public_submissin.csv")
ethalon_df = pd.read_csv("ethalon_submission.csv")

engine = create_engine("sqlite+pysqlite:///:memory:")
base = declarative_base()
session = scoped_session(sessionmaker(bind=engine))()
metadata = MetaData()

my_table = Table(
    'questions', metadata,
    Column('id', Integer, primary_key=True),
    Column('Unnamed: 0', Integer),
    Column('question', String),
    Column('query', String),
    Column('result', String),
    Column('knowledge', String),
    Column('prompt', String)
)

class Questions(base):
    __table__ = my_table

    def __repr__(self):
        return f"<Question(question='{self.question[:50]}...')>"

# Create the table with the defined schema
metadata.create_all(engine)

# Load data from df into the created table
df.to_sql('questions', engine, if_exists='append', index=False)


def get_by_question(cls, question):
    query = session.query(cls).filter(cls.question == question).first()
    if query is None:
        return ''
    else:
        return query.result

ethalon_df['test_answer'] = ethalon_df['Вопрос'].apply(lambda x: get_by_question(Questions, x))

ethalon_df.to_csv("data_for_comparison.csv")