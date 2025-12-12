import pymysql

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="비밀번호",
        database="pet_ai",
        cursorclass=pymysql.cursors.DictCursor
    )