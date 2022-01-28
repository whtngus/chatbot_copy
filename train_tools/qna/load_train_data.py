import pymysql
import openpyxl

from config.DatabaseConfig import * # DB 접속 정보 불러오기


# 학습 데이터 초기화
def all_clear_train_data(db):
    # 기존 학습 데이터 삭제
    sql = '''
            delete from chatbot_train_data
        '''
    db.execute(sql)

    # auto increment 초기화
    # sql = '''
    # ALTER TABLE chatbot_train_data AUTOINCREMENT=1
    # '''
    # db.execute(sql)


# db에 데이터 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_img_url = xls_row

    sql = '''
        INSERT INTO chatbot_train_data (intent, ner, query, answer, answer_image) 
        values(
         '%s', '%s', '%s', '%s', '%s'
        )
    ''' % (intent.value, ner.value, query.value, answer.value, answer_img_url.value)

    # 엑셀에서 불러온 cell에 데이터가 없는 경우, null 로 치환
    sql = sql.replace("'None'", "null")

    # with db.cursor() as cursor:
    db.execute(sql)
    print('{} 저장'.format(query.value))


train_file = './train_data.xlsx'
db = None
try:
    import sqlite3
    conn = sqlite3.connect("../../chatbot.db")
    db = conn.cursor()

    # 기존 학습 데이터 초기화
    all_clear_train_data(db)

    # 학습 엑셀 파일 불러오기
    wb = openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']
    for row in sheet.iter_rows(min_row=2): # 해더는 불러오지 않음
        # 데이터 저장
        insert_data(conn, row)

    wb.close()
    conn.commit()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()

