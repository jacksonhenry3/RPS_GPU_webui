import atexit
import re
import tempfile
import os 

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

from web.handlers import register_handlers
from web.state import rps_sim, sim_state, nvimgcodec_encoder

# --- Temporary File Management ---
temp_dir = tempfile.TemporaryDirectory()
atexit.register(temp_dir.cleanup)

# --- Flask & SocketIO Initialization ---
# Determine absolute paths for templates and static folders
web_dir = os.path.dirname(os.path.abspath(__file__))
page_dir = os.path.join(web_dir, 'page')
template_folder = os.path.join(page_dir, 'templates')
static_folder = os.path.join(page_dir, 'static')

app = Flask(
    __name__,
    template_folder=template_folder,
    static_folder=static_folder
)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', binary=True)

# --- WSGI Middleware for Reverse Proxy ---
class OODProxyMiddleware:
    def __init__(self, app):
        self.app = app
        self.proxy_pattern = re.compile(r'^(/(?:node|rnode)/[^/]+/[^/]+)(.*)')

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        match = self.proxy_pattern.match(path_info)
        if match:
            script_name, app_path = match.groups()
            environ['SCRIPT_NAME'] = script_name
            environ['PATH_INFO'] = app_path if app_path else '/'
        return self.app(environ, start_response)

app.wsgi_app = OODProxyMiddleware(app.wsgi_app)

# --- Register Handlers ---
register_handlers(socketio, app, rps_sim, sim_state, nvimgcodec_encoder, temp_dir)

# --- Main Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(temp_dir.name, filename, as_attachment=True)

@app.route('/plots/<filename>')
def plots(filename):
    return send_from_directory(temp_dir.name, filename)
