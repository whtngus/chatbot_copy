import pymysql
from config.DatabaseConfig import * # DB 접속 정보 불러오기

db = None
try:
    import sqlite3

    conn = sqlite3.connect("../../chatbot.db")
    cursor = conn.cursor()
    sql = """
    drop table chatbot_train_data
    """
    cursor.execute(sql)

    # 테이블 생성 sql 정의
    sql = '''
      CREATE TABLE chatbot_train_data (
      `id` INTEGER   PRIMARY KEY AUTOINCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      `answer_image` VARCHAR(2048) NULL)
    '''

    # 테이블 생성
    # with db.cursor() as cursor:
    cursor.execute(sql)

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()

