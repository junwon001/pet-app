from db import get_connection

def save_bcs(pet_id: int, bcs_value: int):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO bcs_history (pet_id, bcs_value)
            VALUES (%s, %s)
            """
            cursor.execute(sql, (pet_id, bcs_value))
        conn.commit()
    finally:
        conn.close()
def get_latest_bcs(pet_id: int):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT bcs_value
            FROM bcs_history
            WHERE pet_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """
            cursor.execute(sql, (pet_id,))
            row = cursor.fetchone()
            return row["bcs_value"] if row else None
    finally:
        conn.close()

