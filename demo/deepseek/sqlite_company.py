import sqlite3
import json


# 初始化数据库并插入测试数据
def init_database():
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()

    # 创建员工表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary REAL,
            is_active INTEGER
        )
    ''')

    # 插入示例数据
    employees = [
        ('张三', '技术部', 12000, 1),
        ('李四', '技术部', 9000, 1),
        ('王五', '市场部', 8500, 1),
        ('赵六', '财务部', 7500, 0)
    ]
    cursor.executemany(
        'INSERT INTO employees (name, department, salary, is_active) VALUES (?, ?, ?, ?)',
        employees
    )

    conn.commit()
    conn.close()


if __name__ == '__main__':
    # 执行初始化
    init_database()
