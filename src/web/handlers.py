import os
from flask import request
from eventlet.event import Event
from nvidia import nvimgcodec

import cupy as cp

from app_utils import log_server, log_separator
from web.threads import (
    simulation_loop,
    render_loop,
    process_recording,
    _emit_frame,
    _generate_filename,
)
from plotting import generate_plots

def register_handlers(socketio, app, rps_sim, sim_state, nvimgcodec_encoder, temp_dir):
    def stop_simulation_threads():
        sim_state.is_running = False
        # Unblock any waiting threads
        if not sim_state.client_ready.ready():
            sim_state.client_ready.send()
        if not sim_state.sync_event_render_ready.ready():
            sim_state.sync_event_render_ready.send()
        if not sim_state.sync_event_sim_done.ready():
            sim_state.sync_event_sim_done.send()
        # Join threads
        if sim_state.sim_thread:
            sim_state.sim_thread.join()
            sim_state.sim_thread = None
        if sim_state.render_thread:
            sim_state.render_thread.join()
            sim_state.render_thread = None

    @socketio.on('set_sync_mode')
    def set_sync_mode(data):
        is_sync = data.get('is_synchronized', False)
        sim_state.is_synchronized = is_sync
        log_server(f"Synchronization mode set to: {is_sync}")

        if sim_state.is_running:
            stop_simulation_threads()
            handle_start()

    @socketio.on('client_ready_for_next_frame')
    def handle_client_ready():
        if not sim_state.client_ready.ready():
            sim_state.client_ready.send()

    @socketio.on('start_simulation')
    def handle_start():
        if not sim_state.is_running:
            log_server("Start command received. Starting simulation threads.")

            sim_state.client_ready = Event()
            sim_state.sync_event_sim_done = Event()
            sim_state.sync_event_render_ready = Event()
            sim_state.client_ready.send()

            sim_state.is_running = True
            sim_state.is_plotting = True

            sim_state.perf['sim_steps_ps'] = 0
            sim_state.perf['render_fps'] = 0
            sim_state.sim_thread = socketio.start_background_task(
                target=simulation_loop, socketio=socketio, rps_sim=rps_sim, sim_state=sim_state
            )
            sim_state.render_thread = socketio.start_background_task(
                target=render_loop,
                socketio=socketio,
                rps_sim=rps_sim,
                sim_state=sim_state,
                nvimgcodec_encoder=nvimgcodec_encoder,
            )

    @socketio.on('pause_simulation')
    def handle_pause():
        if sim_state.is_running:
            log_server("Pause command received. Halting simulation.")
            stop_simulation_threads()
            socketio.emit('request_plot_update')

    @socketio.on('reset_simulation')
    def handle_reset():
        log_server("Reset command received.")
        if sim_state.is_running: stop_simulation_threads()
        log_separator()
        sim_state.perf['sim_steps_ps'] = 0
        sim_state.perf['render_fps'] = 0
        sim_state.is_plotting = True # Enable plotting on reset
        sim_state.history_pop = []
        sim_state.history_entropy = []
        sim_state.plot_paths = []
        rps_sim.reset()
        rps_sim.render()
        _emit_frame(socketio, rps_sim, sim_state, nvimgcodec_encoder)
        socketio.emit('request_plot_update')

    @socketio.on('disconnect')
    def handle_disconnect():
        if sim_state.is_running:
            log_server("Client disconnected, stopping simulation.")
            stop_simulation_threads()

    @socketio.on('update_params')
    def handle_update_params(params):
        log_server(f"Received params update from client: {params.get('networkType')}, {params.get('numAgents')} agents")
        was_running = sim_state.is_running
        if was_running: stop_simulation_threads()

        if rps_sim.needs_reinitialization(params):
            log_server("Network change detected, re-initializing simulation...")
            rps_sim.initialize(params)
            socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})

        rps_sim.update_parameters(params)
        rps_sim.render()
        _emit_frame(socketio, rps_sim, sim_state, nvimgcodec_encoder)

        if was_running:
            handle_start()

    @socketio.on('save_frame')
    def handle_save_frame():
        log_server("Save Frame command received.")
        filename = _generate_filename(rps_sim.params, 'png')
        filepath = os.path.join(temp_dir.name, filename)
        image_array_gpu = rps_sim.visualizer.image_gpu
        nv_image = nvimgcodec.as_image(image_array_gpu)
        png_bytes = nvimgcodec_encoder.encode(nv_image, "png")
        with open(filepath, 'wb') as f: f.write(png_bytes)
        hostname = os.environ.get('HOSTNAME')
        if hostname and (hostname.startswith('gpu-') or '.' in hostname):
            base_hostname = hostname.split('.')[0]
            full_hostname = f"{base_hostname}.cm.cluster"
            proxy_prefix = f"/node/{full_hostname}/{os.environ['PORT']}"
        else:
            proxy_prefix = f'http://localhost:{os.environ["PORT"]}'
        
        download_url = f"{proxy_prefix}/download/{filename}"
        print(download_url)
        log_server(f"✅ Frame saved for download: {filename}")
        socketio.emit('frame_saved', {'url': download_url, 'filename': filename})

    @socketio.on('start_recording')
    def handle_start_recording():
        if not sim_state.is_recording:
            log_server("🔴 Recording started.")
            sim_state.is_recording = True
            nv_image = nvimgcodec.as_image(rps_sim.visualizer.image_gpu.astype(cp.uint8))
            png_bytes = nvimgcodec_encoder.encode(nv_image, "png")
            sim_state.recorded_frames = [png_bytes]
            sim_state.proxy_prefix = request.environ.get('SCRIPT_NAME', '')

    @socketio.on('stop_recording')
    def handle_stop_recording(proxy_prefix=None):
        if sim_state.is_recording:
            log_server("⏹️ Recording stopped. Processing video...")
            sim_state.is_recording = False
            if proxy_prefix is None:
                proxy_prefix = request.environ.get('SCRIPT_NAME', '')
            socketio.start_background_task(process_recording, socketio, rps_sim, sim_state, temp_dir, proxy_prefix=proxy_prefix)

    @socketio.on('step_simulation')
    def handle_step():
        if sim_state.is_running: return
        log_server("Step command received.")
        rps_sim.step()
        rps_sim.render()
        _emit_frame(socketio, rps_sim, sim_state, nvimgcodec_encoder)
        socketio.emit('request_plot_update')

    @socketio.on('connect')
    def handle_connect():
        log_server("Client connected.")
        socketio.emit('simulation_config', {'height': rps_sim.visualizer.HEIGHT})
        rps_sim.render()
        _emit_frame(socketio, rps_sim, sim_state, nvimgcodec_encoder)
        # socketio.emit('request_plot_update')


    @app.route('/render_plots')
    def render_plots():
        log_server("Render Plots command received.")
        # sim_state.is_plotting = False # Keep plotting enabled
        plot_paths = generate_plots(sim_state, rps_sim.params, temp_dir.name)
        sim_state.plot_paths = plot_paths
        socketio.emit('plots_ready', {'plot_paths': plot_paths})
        return "Plots rendered!"