import os
import sys

# �T�O Backend �ؿ��b sys.path ��
sys.path.append(os.path.join(os.path.dirname(__file__), 'Backend'))

from server import server

if __name__ == '__main__':
    instance = server()
    instance.call()
