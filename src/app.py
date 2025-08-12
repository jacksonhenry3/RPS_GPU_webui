import os
import re
import uuid
import base64
import argparse
import tempfile
import atexit
import imageio.v2 as imageio
import time
from collections import deque
from cupyx.scipy.ndimage import zoom
import cupy as cp

from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
from nvidia import nvimgcodec
from eventlet.event import Event

from simulation import Simulation
from app_utils import log_server, log_sim, log_error, log_progress, log_separator
from plotting import create_strategy_plot, create_neighbor_plot

# --- Temporary File Management ---
temp_dir = tempfile.TemporaryDirectory()
atexit.register(temp_dir.cleanup)

# --- Flask & SocketIO Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', binary=True)

# --- WSGI Middleware for Reverse Proxy (for HPC environments) ---
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

# --- Global State and Initialization ---
sim_state = {
    'is_running': False,
    'sim_thread': None,
    'render_thread': None,
    'is_recording': False,
    'recorded_frames': [],
    'proxy_prefix': '',
    'is_plotting': False,
    'client_ready': Event(),
    'is_synchronized': False,
    'sync_event_sim_done': Event(),
    'sync_event_render_ready': Event(),
    'perf': {
        'sim_steps_ps': 0,
        'render_fps': 0,
    }
}
rps_sim = Simulation()
nvimgcodec_encoder = nvimgcodec.Encoder()

def _generate_filename(params, extension):
    net = params['networkType'].replace('_', '-')
    size = params['numAgents']
    kt = params['kT']
    hist_val = params.get('historyLength', 1000001)
    hist = 'inf' if hist_val > 1000 else hist_val
    win, tie, loss = params['win'], params['tie'], params['loss']
    return f"net-{net}_s-{size}_kT-{kt:.1f}_hist-{hist}_p-{win}-{tie}-{loss}_{uuid.uuid4().hex[:6]}.{extension}"

def _emit_frame(sim_instance):
    timings = {}
    image_array_gpu = sim_instance.visualizer.image_gpu
    target_width = sim_instance.params.get('renderResolution', 256)
    quality = sim_instance.params.get('jpegQuality', 99)
    original_height, original_width, _ = image_array_gpu.shape

    scale_start = time.time()
    if original_width != target_width:
        scale_factor = target_width / original_width
        image_array_gpu = zoom(image_array_gpu, (scale_factor, scale_factor, 1), order=0)
    timings['scale'] = time.time() - scale_start

    encode_start = time.time()
    nv_image = nvimgcodec.as_image(image_array_gpu.astype(cp.uint8))
    jpeg_bytes = nvimgcodec_encoder.encode(nv_image, "jpeg", params=nvimgcodec.EncodeParams(quality=quality))
    timings['encode'] = time.time() - encode_start

    if sim_state['is_recording']:
        sim_state['recorded_frames'].append(jpeg_bytes)

    emit_start_time = time.time()
    socketio.emit('new_frame', jpeg_bytes)
    socketio.emit('frame_number_update', {'frame_number': sim_instance.time_step})
    timings['emit'] = time.time() - emit_start_time
    return timings

# --- Background Threads ---
def simulation_loop():
    log_server("Starting simulation loop.")
    perf = sim_state['perf']
    last_update_time = time.time()
    steps_since_last_update = 0

    while sim_state['is_running']:
        if sim_state['is_synchronized']:
            sim_state['sync_event_render_ready'].wait()
            sim_state['sync_event_render_ready'] = Event()
            if not sim_state['is_running']: break

            steps_to_run = rps_sim.steps_per_frame
            for _ in range(steps_to_run):
                if not sim_state['is_running']: break
                rps_sim.step()
            
            sim_state['sync_event_sim_done'].send()
        else:
            rps_sim.step()
            steps_since_last_update += 1
            current_time = time.time()
            delta_time = current_time - last_update_time
            if delta_time >= 0.5:
                perf['sim_steps_ps'] = steps_since_last_update / delta_time
                last_update_time = current_time
                steps_since_last_update = 0
            socketio.sleep(0)
    log_server("Simulation loop stopped.")

def render_loop():
    log_server("Starting render loop.")
    perf = sim_state['perf']
    render_times = deque(maxlen=100)
    scale_times = deque(maxlen=100)
    encode_times = deque(maxlen=100)
    emit_times = deque(maxlen=100)
    wait_times = deque(maxlen=100)
    frame_times = deque(maxlen=100)

    last_perf_update = time.time()
    perf_update_frames = 0

    while sim_state['is_running']:
        loop_start_time = time.time()

        wait_start = time.time()
        sim_state['client_ready'].wait()
        if not sim_state['is_running']: break
        wait_time = time.time() - wait_start
        sim_state['client_ready'] = Event()

        if sim_state['is_synchronized']:
            sim_state['sync_event_render_ready'].send()
            sim_state['sync_event_sim_done'].wait()
            sim_state['sync_event_sim_done'] = Event()
            if not sim_state['is_running']: break

        render_start = time.time()
        rps_sim.render()
        render_time = time.time() - render_start

        emit_timings = _emit_frame(rps_sim)

        render_times.append(render_time)
        scale_times.append(emit_timings['scale'])
        encode_times.append(emit_timings['encode'])
        emit_times.append(emit_timings['emit'])
        wait_times.append(wait_time)
        frame_times.append(time.time() - loop_start_time)
        perf_update_frames += 1

        current_time = time.time()
        if current_time - last_perf_update >= 1.0:
            delta_time = current_time - last_perf_update
            render_fps = perf_update_frames / delta_time
            perf['render_fps'] = render_fps

            def avg_ms(q):
                return (sum(q) / len(q)) * 1000 if q else 0

            socketio.emit('perf_update', {
                'sim_fps': f"{perf.get('sim_steps_ps', 0):.0f}",
                'render_fps': f"{perf.get('render_fps', 0):.1f}",
                'frame': f"{avg_ms(frame_times):.2f}",
                'wait': f"{avg_ms(wait_times):.2f}",
                'render': f"{avg_ms(render_times):.2f}",
                'scale': f"{avg_ms(scale_times):.2f}",
                'encode': f"{avg_ms(encode_times):.2f}",
                'emit': f"{avg_ms(emit_times):.2f}",
            })

            last_perf_update = current_time
            perf_update_frames = 0
            render_times.clear(); scale_times.clear(); encode_times.clear(); emit_times.clear(); wait_times.clear(); frame_times.clear()

    log_server("Render loop stopped.")

def process_recording(proxy_prefix=''):
    log_server(f"🎬 Starting video processing for {len(sim_state['recorded_frames'])} frames...")
    if not sim_state['recorded_frames']:
        log_error("No frames to process for recording.")
        socketio.emit('recording_ready', {'url': None, 'error': 'No frames were recorded.'})
        return

    filename = _generate_filename(rps_sim.params, 'mp4')
    filepath = os.path.join(temp_dir.name, filename)

    try:
        with imageio.get_writer(filepath, fps=30, macro_block_size=1) as writer:
            for frame_bytes in sim_state['recorded_frames']:
                writer.append_data(imageio.imread(frame_bytes))

        sim_state['recorded_frames'].clear()
        download_url = f"{proxy_prefix}/download/{filename}"
        log_server(f"✅ Video ready for download: {filename}")
        socketio.emit('recording_ready', {'url': download_url, 'filename': filename})
    except Exception as e:
        log_error(f"Error processing video: {e}")
        socketio.emit('recording_ready', {'url': None, 'error': str(e)})

# --- Socket.IO Handlers ---
@socketio.on('set_sync_mode')
def set_sync_mode(data):
    is_sync = data.get('is_synchronized', False)
    sim_state['is_synchronized'] = is_sync
    log_server(f"Synchronization mode set to: {is_sync}")

    # To prevent any race conditions or deadlocks, we simply restart the
    # simulation threads when the mode is changed while running.
    if sim_state['is_running']:
        stop_simulation_threads()
        handle_start()

@socketio.on('client_ready_for_next_frame')
def handle_client_ready():
    if not sim_state['client_ready'].ready():
        sim_state['client_ready'].send()

@socketio.on('start_simulation')
def handle_start():
    if not sim_state['is_running']:
        log_server("Start command received. Starting simulation threads.")

        # Reset all events to a clean, known state before starting the loops.
        sim_state['client_ready'] = Event()
        sim_state['sync_event_sim_done'] = Event()
        sim_state['sync_event_render_ready'] = Event()
        
        # Manually trigger the client_ready event to kick off the render_loop,
        # since the client is already connected and won't send a new ready signal.
        sim_state['client_ready'].send()

        sim_state['is_running'] = True
        sim_state['perf']['sim_steps_ps'] = 0
        sim_state['perf']['render_fps'] = 0
        sim_state['sim_thread'] = socketio.start_background_task(target=simulation_loop)
        sim_state['render_thread'] = socketio.start_background_task(target=render_loop)

def stop_simulation_threads():
    sim_state['is_running'] = False
    # Unblock any waiting threads
    if not sim_state['client_ready'].ready(): sim_state['client_ready'].send()
    if not sim_state['sync_event_render_ready'].ready(): sim_state['sync_event_render_ready'].send()
    if not sim_state['sync_event_sim_done'].ready(): sim_state['sync_event_sim_done'].send()
    # Join threads
    if sim_state['sim_thread']: sim_state['sim_thread'].join(); sim_state['sim_thread'] = None
    if sim_state['render_thread']: sim_state['render_thread'].join(); sim_state['render_thread'] = None

@socketio.on('pause_simulation')
def handle_pause():
    if sim_state['is_running']:
        log_server("Pause command received. Halting simulation.")
        stop_simulation_threads()
        socketio.emit('request_plot_update')

@socketio.on('reset_simulation')
def handle_reset():
    log_server("Reset command received.")
    if sim_state['is_running']: stop_simulation_threads()
    log_separator()
    sim_state['perf']['sim_steps_ps'] = 0
    sim_state['perf']['render_fps'] = 0
    rps_sim.reset()
    rps_sim.render()
    _emit_frame(rps_sim)
    socketio.emit('request_plot_update')

@socketio.on('disconnect')
def handle_disconnect():
    if sim_state['is_running']:
        log_server("Client disconnected, stopping simulation.")
        stop_simulation_threads()

# --- Other handlers ... ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(temp_dir.name, filename, as_attachment=True)

@socketio.on('update_params')
def handle_update_params(params):
    log_server(f"Received params update from client: {params.get('networkType')}, {params.get('numAgents')} agents")
    was_running = sim_state['is_running']
    if was_running: stop_simulation_threads()
    
    if rps_sim.needs_reinitialization(params):
        log_server("Network change detected, re-initializing simulation...")
        rps_sim.initialize(params)
        socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})
    
    rps_sim.update_parameters(params)
    rps_sim.render()
    _emit_frame(rps_sim)

    if was_running:
        handle_start()

@socketio.on('save_frame')
def handle_save_frame():
    log_server("Save Frame command received.")
    filename = _generate_filename(rps_sim.params, 'jpg')
    filepath = os.path.join(temp_dir.name, filename)
    image_array_gpu = rps_sim.visualizer.image_gpu
    nv_image = nvimgcodec.as_image(image_array_gpu)
    jpeg_bytes = nvimgcodec_encoder.encode(nv_image, "jpeg", params=nvimgcodec.EncodeParams(quality=98))
    with open(filepath, 'wb') as f: f.write(jpeg_bytes)
    proxy_prefix = request.environ.get('SCRIPT_NAME', '')
    download_url = f"{proxy_prefix}/download/{filename}"
    log_server(f"✅ Frame saved for download: {filename}")
    socketio.emit('frame_saved', {'url': download_url, 'filename': filename})

@socketio.on('start_recording')
def handle_start_recording():
    if not sim_state['is_recording']:
        log_server("🔴 Recording started.")
        sim_state['is_recording'] = True
        sim_state['recorded_frames'] = []
        sim_state['proxy_prefix'] = request.environ.get('SCRIPT_NAME', '')

@socketio.on('stop_recording')
def handle_stop_recording(proxy_prefix=None):
    if sim_state['is_recording']:
        log_server("⏹️ Recording stopped. Processing video...")
        sim_state['is_recording'] = False
        if proxy_prefix is None:
            proxy_prefix = request.environ.get('SCRIPT_NAME', '')
        socketio.start_background_task(process_recording, proxy_prefix=proxy_prefix)

@socketio.on('step_simulation')
def handle_step():
    if sim_state['is_running']: return
    log_server("Step command received.")
    rps_sim.step()
    rps_sim.render()
    _emit_frame(rps_sim)
    socketio.emit('request_plot_update')

@socketio.on('connect')
def handle_connect():
    log_server("Client connected.")
    socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})
    rps_sim.render()
    _emit_frame(rps_sim)
    socketio.emit('request_plot_update')

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask-SocketIO RPS Simulation.")
    parser.add_argument('port', type=int, nargs='?', default=4629, help='Port to run the web server on (default: 5000)')
    args = parser.parse_args()
    HOST, PORT = '0.0.0.0', args.port
    os.environ['PORT'] = str(PORT)
    hostname = os.environ.get('HOSTNAME')
    BLUE_BOLD, RESET = '\033[1;94m', '\033[0m'
    log_separator()
    log_server(f"Starting Flask-SocketIO server on {HOST}:{PORT}")
    log_server(f"Temporary file storage is at: {temp_dir.name}")
    if hostname and (hostname.startswith('gpu-') or '.' in hostname):
        full_hostname = f"{hostname}.cm.cluster" if '.' not in hostname else hostname
        node_url = f"https://ondemand.turing.wpi.edu/node/{full_hostname}/{PORT}"
        print(f"✅ On-demand node detected. Connect using: {BLUE_BOLD}{node_url}{RESET}")
    else:
        local_url = f"http://localhost:{PORT}"
        print(f"✅ Access locally at {BLUE_BOLD}{local_url}{RESET}")
    log_separator()
    socketio.run(app, host=HOST, port=PORT)