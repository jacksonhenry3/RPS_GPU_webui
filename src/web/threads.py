import os
import time
import uuid
from collections import deque

import cupy as cp
import imageio.v2 as imageio
from cupyx.scipy.ndimage import zoom
from eventlet.event import Event
from nvidia import nvimgcodec

from app_utils import log_error, log_server


# --- Helper Functions ---
def _generate_filename(params, extension):
    net = params['networkType'].replace('_', '-')
    size = params['numAgents']
    kt = params['kT']
    hist_val = params.get('historyLength', 1000001)
    hist = 'inf' if hist_val > 1000 else hist_val
    win, tie, loss = params['win'], params['tie'], params['loss']
    return f"net-{net}_s-{size}_kT-{kt:.1f}_hist-{hist}_p-{win}-{tie}-{loss}_{uuid.uuid4().hex[:6]}.{extension}"


# The function signature remains the same as your original
def _emit_frame(socketio, sim_instance, sim_state, nvimgcodec_encoder):
    timings = {}
    image_array_gpu = sim_instance.visualizer.image_gpu
    target_width = sim_instance.params.get('renderResolution', 128)
    # The 'jpegQuality' parameter is no longer used for lossless PNG
    # quality = sim_instance.params.get('jpegQuality', 99) 
    original_height, original_width, _ = image_array_gpu.shape

    scale_start = time.time()
    if original_width != target_width:
        # scale_factor = target_width / original_width
        # image_array_gpu = zoom(image_array_gpu, (scale_factor, scale_factor, 1), order=0)
        scale_factor = int(original_width / target_width)
        # print(scale_factor,original_width,target_width)
        image_array_gpu = image_array_gpu[::scale_factor, ::scale_factor, :].copy()
    timings['scale'] = time.time() - scale_start

    encode_start = time.time()
    nv_image = nvimgcodec.as_image(image_array_gpu.astype(cp.uint8))
    
    # --- KEY CHANGE: Encode to PNG instead of JPEG ---
    png_bytes = nvimgcodec_encoder.encode(nv_image, "png")
    
    timings['encode'] = time.time() - encode_start

    if sim_state['is_recording']:
        # Store the lossless PNG bytes for video creation
        sim_state['recorded_frames'].append(png_bytes)

    emit_start_time = time.time()
    # Emit the PNG bytes; all modern browsers support PNG natively
    socketio.emit('new_frame', png_bytes)
    socketio.emit('frame_number_update', {'frame_number': sim_instance.time_step})
    timings['emit'] = time.time() - emit_start_time
    
    return timings


# --- Background Threads ---
def simulation_loop(socketio, rps_sim, sim_state):
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


def render_loop(socketio, rps_sim, sim_state, nvimgcodec_encoder):
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

        emit_timings = _emit_frame(socketio, rps_sim, sim_state, nvimgcodec_encoder)

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



def process_recording(socketio, rps_sim, sim_state, temp_dir, proxy_prefix=''):
    """
    Processes recorded frames and saves them as an animated GIF.
    """
    log_server(f"🎬 Starting GIF processing for {len(sim_state['recorded_frames'])} frames...")
    if not sim_state['recorded_frames']:
        log_error("No frames to process for recording.")
        socketio.emit('recording_ready', {'url': None, 'error': 'No frames were recorded.'})
        return

    # Change the filename extension from 'mp4' to 'gif'
    filename = _generate_filename(rps_sim.params, 'gif')
    filepath = os.path.join(temp_dir.name, filename)

    try:
        # Use imageio's GIF writer. Parameters are simpler than for video.
        # 'mode=I' processes each frame individually. 'loop=0' creates an infinite loop.
        with imageio.get_writer(filepath, mode='I', fps=15, loop=0) as writer:
            for frame_bytes in sim_state['recorded_frames']:
                # imageio.imread can decode the in-memory PNG bytes
                writer.append_data(imageio.imread(frame_bytes))
        
        sim_state['recorded_frames'].clear()
        download_url = f"{proxy_prefix}/download/{filename}"
        log_server(f"✅ GIF ready for download: {filename}")
        socketio.emit('recording_ready', {'url': download_url, 'filename': filename})
        
    except Exception as e:
        log_error(f"Error processing GIF: {e}")
        socketio.emit('recording_ready', {'url': None, 'error': str(e)})
