import argparse
import os
import sys
import socket

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app_utils import log_server, log_separator
from web.app import app, socketio
parser = argparse.ArgumentParser(description="Flask-SocketIO RPS Simulation.")
parser.add_argument('port', type=int, nargs='?', default=4629, help='Port to run the web server on (default: 5000)')
args = parser.parse_args()
HOST, PORT = '0.0.0.0', args.port
os.environ['PORT'] = str(PORT)

hostname = socket.getfqdn().split('.')[0]

BLUE_BOLD, RESET = '\033[1;94m', '\033[0m'
log_separator()
log_server(f"Starting Flask-SocketIO server on {HOST}:{PORT}")
full_hostname = f"{hostname}.int.turing.wpi.edu" if '.' not in hostname else hostname
node_url = f"https://ondemand.turing.wpi.edu/node/{full_hostname}/{PORT}"
print(f"✅ On-demand node detected. Connect using: {BLUE_BOLD}{node_url}{RESET}")
log_separator()
socketio.run(app, host=HOST, port=PORT)
