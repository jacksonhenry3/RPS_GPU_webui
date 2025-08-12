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

from simulation import Simulation
from app_utils import log_server, log_sim, log_error, log_progress, log_separator
from plotting import create_strategy_plot, create_neighbor_plot

# --- Temporary File Management ---
temp_dir = tempfile.TemporaryDirectory()
atexit.register(temp_dir.cleanup)

# --- Flask & SocketIO Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', binary=True) # Enable binary support

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
    'thread': None,
    'is_recording': False,
    'recorded_frames': [],
    'proxy_prefix': '',
    'is_plotting': False,
    'perf': {
        'sim_times': deque(maxlen=100),
        'render_times': deque(maxlen=100),
        'scale_times': deque(maxlen=100),
        'encode_times': deque(maxlen=100),
        'emit_times': deque(maxlen=100),
        'wait_times': deque(maxlen=100), # New: For timing socketio.sleep()
        'other_times': deque(maxlen=100),
        'frame_times': deque(maxlen=100),
        'last_update': time.time(),
        'fps_frames': 0
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
    """Encodes and emits the current simulation frame, returning detailed timings."""
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
    jpeg_bytes = nvimgcodec_encoder.encode(
        nv_image, "jpeg", params=nvimgcodec.EncodeParams(quality=quality)
    )
    timings['encode'] = time.time() - encode_start

    if sim_state['is_recording']:
        sim_state['recorded_frames'].append(jpeg_bytes)

    emit_start_time = time.time()
    # Send raw bytes directly. The `binary=True` flag in the SocketIO constructor is important.
    socketio.emit('new_frame', jpeg_bytes)
    socketio.emit('frame_number_update', {'frame_number': sim_instance.time_step})
    timings['emit'] = time.time() - emit_start_time

    return timings

# --- Background Simulation Thread ---
def simulation_loop():
    perf = sim_state['perf']

    while sim_state['is_running']:
        loop_start_time = time.time()

        # --- 1. Simulation Step ---
        sim_start_time = time.time()
        steps_to_run = rps_sim.steps_per_frame
        for _ in range(steps_to_run):
            if not sim_state['is_running']: break
            rps_sim.step()
        if not sim_state['is_running']: break
        sim_time = time.time() - sim_start_time

        # --- 2. Render Step ---
        render_start_time = time.time()
        rps_sim.render()
        render_time = time.time() - render_start_time

        # --- 3. Encode & Emit Step ---
        emit_timings = _emit_frame(rps_sim)

        # --- 4. Wait Step ---
        wait_start_time = time.time()
        socketio.sleep(0) # Sleep to prevent pegging the CPU
        wait_time = time.time() - wait_start_time

        # --- 5. Performance Calculation ---
        loop_end_time = time.time()
        total_frame_time = loop_end_time - loop_start_time
        
        # Calculate 'other' time as the remainder
        known_time = (sim_time + render_time + emit_timings['scale'] +
                      emit_timings['encode'] + emit_timings['emit'] + wait_time)
        other_time = total_frame_time - known_time

        # Aggregate performance data
        perf['sim_times'].append(sim_time)
        perf['render_times'].append(render_time)
        perf['scale_times'].append(emit_timings['scale'])
        perf['encode_times'].append(emit_timings['encode'])
        perf['emit_times'].append(emit_timings['emit'])
        perf['wait_times'].append(wait_time)
        perf['other_times'].append(other_time)
        perf['frame_times'].append(total_frame_time)
        perf['fps_frames'] += 1

        # --- 6. Emit Performance Update (Periodically) ---
        current_time = time.time()
        if current_time - perf['last_update'] >= 1.0:
            frame_count = len(perf['frame_times'])
            if frame_count > 0:
                # Helper to calculate average time in ms
                def avg_ms(times):
                    return (sum(times) / len(times)) * 1000

                fps = perf['fps_frames'] / (current_time - perf['last_update'])

                socketio.emit('perf_update', {
                    'fps': round(fps),
                    'frame': f"{avg_ms(perf['frame_times']):.2f}",
                    'sim': f"{avg_ms(perf['sim_times']):.2f}",
                    'render': f"{avg_ms(perf['render_times']):.2f}",
                    'scale': f"{avg_ms(perf['scale_times']):.2f}",
                    'encode': f"{avg_ms(perf['encode_times']):.2f}",
                    'emit': f"{avg_ms(perf['emit_times']):.2f}",
                    'wait': f"{avg_ms(perf['wait_times']):.2f}",
                    'other': f"{avg_ms(perf['other_times']):.2f}"
                })

            perf['last_update'] = current_time
            perf['fps_frames'] = 0
            # Clear all timing deques for the next interval
            for key in list(perf.keys()):
                if key.endswith('_times'):
                    perf[key].clear()

    log_server("Simulation loop stopped.")

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

# --- Flask & Socket.IO Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(temp_dir.name, filename, as_attachment=True)

def generate_and_send_plots():
    if sim_state.get('is_plotting', False): return
    try:
        sim_state['is_plotting'] = True
        strategy_img_bytes = create_strategy_plot(rps_sim.strategy_history)
        neighbor_img_bytes = create_neighbor_plot(rps_sim.neighbor_history)
        socketio.emit('plot_update', {
            'strategy_plot': base64.b64encode(strategy_img_bytes).decode('utf-8'),
            'neighbor_plot': base64.b64encode(neighbor_img_bytes).decode('utf-8'),
        })
    finally:
        sim_state['is_plotting'] = False

# --- Socket.IO Handlers ---
@socketio.on('request_plot_update')
def handle_request_plot_update():
    pass
    # socketio.start_background_task(generate_and_send_plots)

@socketio.on('update_params')
def handle_update_params(params):
    log_server(f"Received params update from client: {params.get('networkType')}, {params.get('numAgents')} agents")
    if rps_sim.needs_reinitialization(params):
        log_server("Network change detected, re-initializing simulation...")
        sim_state['is_running'] = False
        rps_sim.initialize(params)
        socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})
    rps_sim.update_parameters(params)

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

@socketio.on('start_simulation')
def handle_start():
    if not sim_state['is_running']:
        log_server("Start command received. Starting simulation thread.")
        sim_state['is_running'] = True
        sim_state['thread'] = socketio.start_background_task(target=simulation_loop)

@socketio.on('pause_simulation')
def handle_pause():
    if sim_state['is_running']:
        log_server("Pause command received. Halting simulation.")
        sim_state['is_running'] = False
        socketio.emit('request_plot_update')

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

@socketio.on('reset_simulation')
def handle_reset():
    log_server("Reset command received.")
    if sim_state['is_running']:
        sim_state['is_running'] = False
        if sim_state['thread']:
            sim_state['thread'].join()
            sim_state['thread'] = None
    log_separator()
    # Reset performance stats
    perf = sim_state['perf']
    perf['last_update'] = time.time()
    for key in list(perf.keys()):
        if key.endswith('_times'):
            perf[key].clear()
    perf['fps_frames'] = 0

    rps_sim.reset()
    _emit_frame(rps_sim)
    socketio.emit('request_plot_update')

@socketio.on('connect')
def handle_connect():
    log_server("Client connected.")
    socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})
    _emit_frame(rps_sim)
    socketio.emit('request_plot_update')


@socketio.on('disconnect')
def handle_disconnect():
    sim_state['is_running'] = False
    log_server("Client disconnected.")

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

