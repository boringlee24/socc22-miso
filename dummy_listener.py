import pdb
import time
import os
import subprocess
import re
import random
import json
import numpy as np
import glob
import socket
import argparse
import threading
import _thread
import signal
import sys
from datetime import datetime

def thread_function(host_node):
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host_node, 10002)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    sock.listen(5)  
    while True:
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    connection.sendall(b'success')
                    print(f'received {str(data)}')
                    print(f'current time {datetime.now().time()}')
                else:
                    break
        finally:
            connection.close()

if __name__ == '__main__':
    thread_function(sys.argv[1])
