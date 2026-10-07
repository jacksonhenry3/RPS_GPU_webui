"""Run simulations without a frontend, or plot completed/partial saved runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def run(args):
    # GPU imports belong only to execution: `plot` and --help need no CUDA/UI.
    import cupy as cp
    from simulation import Simulation
    from entropy import state_labels, measure, bank_labels

    params = json.loads(args.config.read_text()) if args.config else {}
    defaults = Simulation._get_default_params()
    unknown = params.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown parameters: {sorted(unknown)}')
    params = defaults | params
    if params['networkType'] not in ('ring_1d_periodic', 'ring_1d_hard',
                                     'grid_2d_periodic', 'grid_2d_hard',
                                     'grid_2d_moore_periodic', 'grid_2d_moore_hard'):
        raise ValueError('Unsupported networkType; see README')
    if not isinstance(params['numAgents'], int) or params['numAgents'] < 3:
        raise ValueError('numAgents must be an integer >= 3 (side length in 2D)')
    if params['selectionMode'] not in ('local_choice', 'global_random', 'deterministic'):
        raise ValueError('Unknown selectionMode')
    if params['scoreCalculationMode'] not in ('total', 'average'):
        raise ValueError('Unknown scoreCalculationMode')
    if params['bankCondition'] not in ('random', 'constant'):
        raise ValueError('Unknown bankCondition')
    if params['initialCondition'] not in ('random', 'all_rock', 'vertical_stripes', 'pie_slices',
            'single_invader', 'double_invader', 'cross_invasion', 'corner_siege',
            'diamond_lattice', 'periodic_stripes', 'symmetric_gradient', 'split'):
        raise ValueError('Unknown initialCondition')
    for key in ('win', 'tie', 'loss', 'bankValue', 'kT', 'kymoAspect'):
        if not np.isfinite(params[key]):
            raise ValueError(f'{key} must be finite')
    if params['kT'] < 0 or params['kymoAspect'] <= 0 or (params['bankCondition'] == 'random' and params['bankValue'] < 0):
        raise ValueError('kT/random bankValue must be nonnegative; kymoAspect positive')
    bins, edges = params['bankBins'], params['bankEdges']
    bank_labels(np.array([0.0]), bins, edges)  # validate before creating output
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must lie in [0, 2**32)')

    args.out.mkdir(parents=True, exist_ok=False)  # never mix or overwrite runs
    metadata = {'schema_version': 1, 'status': 'running', 'parameters': params,
                'seed': args.seed, 'steps_requested': args.steps, 'steps_completed': 0,
                'sample_every': args.sample_every, 'snapshot_every': args.snapshot_every,
                'temporal': args.temporal, 'temporal_lag': 1,
                'bank_binning': 'fixed_thresholds' if edges is not None else 'per_sample_equal_width',
                'python': platform.python_version(), 'numpy': np.__version__, 'cupy': cp.__version__,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')}
    repo = Path(__file__).resolve().parent
    try:
        metadata['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        metadata['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata['git_commit'] = None

    def save_metadata():
        tmp = args.out / 'metadata.json.tmp'
        tmp.write_text(json.dumps(metadata, indent=2) + '\n')
        tmp.replace(args.out / 'metadata.json')

    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    started = time.perf_counter()
    sim = None
    save_metadata()
    try:
        np.random.seed(args.seed)
        cp.random.seed(args.seed)
        metadata['cuda_runtime'] = cp.cuda.runtime.runtimeGetVersion()
        metadata['cuda_driver'] = cp.cuda.runtime.driverGetVersion()
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        metadata['gpu'] = device.decode() if isinstance(device, bytes) else str(device)
        sim = Simulation(params, visualize=False)
        metadata['num_agents_actual'] = sim.num_agents
        system = sim.agent_system

        def labels():
            return state_labels(system.agent_strategies, system.agent_bank_values, bins, edges, cp)

        def snapshot(name):
            # Atomic replace; these are analysis states, not RNG restart checkpoints.
            with (args.out / (name + '.tmp')).open('wb') as handle:
                np.savez_compressed(handle, step=sim.time_step,
                                    strategies=cp.asnumpy(cp.argmax(system.agent_strategies, axis=0)),
                                    bank=cp.asnumpy(system.agent_bank_values))
            (args.out / (name + '.tmp')).replace(args.out / name)

        snapshot('initial.npz')
        with (args.out / 'measurements.csv').open('w', newline='', buffering=1) as table, \
                (args.out / 'distributions.jsonl').open('w', buffering=1) as distributions:
            writer = None

            def record(previous=None):
                nonlocal writer
                metrics, dist = measure(labels(), bins, previous, cp)
                row = {'step': sim.time_step, 'bank_min': float(system.agent_bank_values.min()),
                       'bank_max': float(system.agent_bank_values.max()), **metrics,
                       'softmax_headroom_nats': system.pop_worst_exp_headroom()}
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                distributions.write(json.dumps({'step': sim.time_step,
                    **{key: cp.asnumpy(value).tolist() for key, value in dist.items()}}) + '\n')
                metadata['steps_completed'] = sim.time_step
                save_metadata()

            record()
            for step in range(1, args.steps + 1):
                if stopped:
                    break
                sample = step % args.sample_every == 0 or step == args.steps
                previous = labels() if args.temporal and sample else None
                sim.step()
                if sample:
                    record(previous)
                if args.snapshot_every and step % args.snapshot_every == 0:
                    snapshot(f'state_{step:012d}.npz')
            if sim.time_step != metadata['steps_completed']:
                record()  # partial final step; no invented temporal interval
        snapshot('final.npz')
        metadata['status'] = 'interrupted' if stopped else 'complete'
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if sim is not None:
            metadata['steps_completed'] = sim.time_step
        metadata['elapsed_seconds'] = time.perf_counter() - started
        save_metadata()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f"{metadata['status']}: {args.out}")
    return 130 if stopped else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser('run', help='GPU simulation with no rendering or web imports')
    runner.add_argument('--config', type=Path, help='JSON simulation parameters; omitted keys use UI defaults')
    runner.add_argument('--out', type=Path, required=True, help='new directory for this run')
    runner.add_argument('--steps', type=positive, required=True)
    runner.add_argument('--seed', type=int, default=31415)
    runner.add_argument('--sample-every', type=positive, default=100)
    runner.add_argument('--snapshot-every', type=int, default=0, help='0 disables intermediate full states')
    runner.add_argument('--temporal', action='store_true', help='measure realized t-1 → t pairs at sample times')
    plotter = commands.add_parser('plot', help='plot saved CSV on a CPU without CUDA or Flask')
    plotter.add_argument('run_dir', type=Path)
    plotter.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plot':
        from plotting import plot_saved_run
        plot_saved_run(args.run_dir, args.out)
        return 0
    if args.snapshot_every < 0:
        parser.error('--snapshot-every must be nonnegative')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
