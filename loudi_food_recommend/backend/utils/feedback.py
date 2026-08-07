from utils.database import get_db_connection
from sqlalchemy import text

# 保存用户反馈
def save_user_feedback(user_id, merchant_id, is_satisfied, reason):
    engine = get_db_connection()
    if engine is None:
        # 数据库不可用时仅打印日志，不阻断流程
        print(f"[feedback] DB unavailable, feedback discarded: user={user_id}, merchant={merchant_id}")
        return

    # 建表（如不存在）- 不加外键约束，避免依赖 ads 表存在
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            merchant_id VARCHAR(50) NOT NULL,
            is_satisfied BOOLEAN NOT NULL,
            reason TEXT,
            feedback_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    insert_sql = text("""
        INSERT INTO feedback (user_id, merchant_id, is_satisfied, reason)
        VALUES (:user_id, :merchant_id, :is_satisfied, :reason)
    """)

    with engine.begin() as conn:
        conn.execute(create_table_sql)
        conn.execute(insert_sql, {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "is_satisfied": is_satisfied,
            "reason": reason,
        })