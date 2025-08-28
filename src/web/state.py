from collections import deque
from eventlet.event import Event
from nvidia import nvimgcodec

from simulation import Simulation

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
