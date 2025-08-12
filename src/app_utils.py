import sys
import time

# ANSI escape codes for colors
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

LINE_CLEAR = '\r' + ' ' * 80 + '\r'

def _log(prefix, color, message):
    """Generic logging function to clear the line and print a formatted message."""
    print(f"{LINE_CLEAR}{color}{Colors.BOLD}[{prefix}]{Colors.ENDC} {message}")

def log_server(message):
    _log("SERVER", Colors.BLUE, message)

def log_sim(message):
    _log("SIM", Colors.GREEN, message)
    
def log_network(message):
    _log("NETWORK", Colors.MAGENTA, message)

def log_error(message):
    _log("ERROR", Colors.RED, message)

def log_progress(message):
    """Prints a progress message on a single line without a newline."""
    print(f"{LINE_CLEAR}{Colors.YELLOW}{message}{Colors.ENDC}")
    # sys.stdout.flush()

def log_separator():
    """Prints a separator to distinguish simulation runs."""
    print("\n" + "-"*70 + "\n")

