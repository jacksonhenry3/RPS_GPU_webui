from collections import deque
from eventlet.event import Event
from nvidia import nvimgcodec

from simulation import Simulation

class State:
    def __init__(self):
        self.is_running = False
        self.sim_thread = None
        self.render_thread = None
        self.is_recording = False
        self.recorded_frames = []
        self.proxy_prefix = ''
        self.is_plotting = False
        self.client_ready = Event()
        self.is_synchronized = False
        self.sync_event_sim_done = Event()
        self.sync_event_render_ready = Event()
        self.perf = {
            'sim_steps_ps': 0,
            'render_fps': 0,
        }
        self.history_pop = []
        self.history_entropy = []
        self.history_appeals = []
        self.plot_paths = []

# --- Global State and Initialization ---
sim_state = State()
rps_sim = Simulation()
nvimgcodec_encoder = nvimgcodec.Encoder()