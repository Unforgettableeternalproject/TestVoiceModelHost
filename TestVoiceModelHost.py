import os
import sys

# 確保 Backend 目錄在 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from server import server

if __name__ == '__main__':
    instance = server()
    instance.call()
