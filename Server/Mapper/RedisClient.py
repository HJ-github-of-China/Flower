"""
1. 封装redis的客户端连接对象
"""
import os

from redis import StrictRedis
from dotenv import load_dotenv

# 创建客户端
load_dotenv()

# 获取环境变量并转换为合适的类型
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_db = os.getenv('REDIS_DB')
redis_password = os.getenv('REDIS_PASSWORD')

# 将 port 和 db 转换为 int 类型，处理 None 值
port = int(redis_port) if redis_port is not None else 6379  # 默认端口 6379
db = int(redis_db) if redis_db is not None else 2  # 默认数据库 0

# 创建客户端
redis_client = StrictRedis(host=redis_host, port=port, db=db, password=redis_password)
