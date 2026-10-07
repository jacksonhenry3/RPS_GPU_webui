# Running the RPS GPU WebUI

## Cluster Setup Instructions

1. **Request a GPU node:**
   ```sh
   sinteractive
   ```

2. **Load required modules:**
   ```sh
   module load uv
   module load cuda12.6
   ```
3. **set environment variables**
   ```sh
   export HOSTNAME
   ```
   
4. **Run the application:**
   ```sh
   uv run run.py [port]
   ```
   `port` is optional and defaults to `4629`.

5. **Access the web interface:**
   - The server prints an on-demand URL to the terminal, e.g.:
     ```
     https://ondemand.turing.wpi.edu/node/gpu-5-28.int.turing.wpi.edu/4629
     ```
     Open it in your browser.



- **AgentSystem**: Contains the core simulation logic for the Rock-Paper-Scissors agent-based model.
- **initial_conditions.py**: Houses standalone functions responsible for setting up the initial states and configurations of the agent system.
- **measurements.py**: Provides functions for calculating various metrics and measurements from the simulation data.
- **web/threads.py**: Manages the background threads for the simulation and rendering, interacting with the `AgentSystem` and utilizing the measurement functions.
- **web/app.py**: The main entry point for the web interface, handling routes and overall application flow.

## Headless GPU runs and offline plots

The batch entry point runs the same `AgentSystem` as the UI, with no Flask,
image codec, rendering, frame history, or kymograph stopping condition. The
simulation still requires CuPy/CUDA; this is not a new CPU physics backend.
The existing `uv run run.py [port]` UI command is unchanged.

```sh
uv sync
uv run batch.py run --config configs/headless.json --steps 100000 \
  --seed 31415 --sample-every 100 --temporal --out runs/example
uv run batch.py plot runs/example --out runs/example/plots
sbatch submit.sh configs/headless.json runs 100000 31415
# Independent repeats: task i uses seed 31415+i and its own output directory.
sbatch --array=0-9 submit.sh configs/headless.json runs 100000 31415
```

Submit from the repository root after creating `.venv`. Adjust the partition,
CUDA module and resource requests in `submit.sh` to your cluster. One GPU task
uses eight CPU threads; this is not an eight-process distributed simulation.
The batch script handles SIGTERM before the time limit and Python flushes the
partial run. SIGKILL/node failure cannot save a final snapshot. Interrupted
jobs return a nonzero exit code and metadata marks them `interrupted`.

To plot on a CPU-only machine, install NumPy and Matplotlib and run
`python batch.py plot RUN_DIR --out FIGURE_DIR`; this command imports neither
CuPy nor the frontend. `plot_measurements(rows, output_dir)` in
`src/plotting.py` is the same renderer used by the UI.

Each output directory must be new. It contains:

| File | Content |
| --- | --- |
| `metadata.json` | Schema, resolved parameters, seed, Git revision/dirty flag, software/GPU versions, Slurm IDs, status and completed steps |
| `measurements.csv` | Streaming scalar measurements at t=0, sampling intervals and the final step |
| `distributions.jsonl` | Per-sample 3×k observed joint probabilities and optional sparse temporal pair probabilities |
| `initial.npz`, `final.npz` | Integer strategy labels (R=0, P=1, S=2), raw bank values and step |
| `state_000000000100.npz` | Optional full states, enabled by `--snapshot-every 100` |

Full states permit new offline measurements without rerunning physics. They
are not exact restart checkpoints: RNG state is not included. Intermediate
snapshots are off by default. Measurement history is streamed, so RAM does
not grow with the number of steps. Disk still grows with samples; temporal
pair output has up to min(N, (3k)²) occupied entries per sample. Seed and
configuration reproduce a run on a compatible software/hardware setup;
bitwise agreement across different GPU/CUDA versions is not promised.

For 2D networks `numAgents` is the side length, so N=`numAgents`². Supported
network types match the UI: `ring_1d_periodic`, `ring_1d_hard`,
`grid_2d_periodic`, `grid_2d_hard`, `grid_2d_moore_periodic`,
`grid_2d_moore_hard`. JSON accepts the existing simulation parameter names,
plus `bankBins` and optional `bankEdges`; unknown names are rejected.
`historyLength` is a legacy UI parameter; the current physics accumulates
bank values without a finite memory window. This patch does not change that.

## Entropy definitions

Choose k=`bankBins` (default 3). The measured state is z=(strategy, bank bin),
with 3k possible values. Bins affect measurements only. By default each
sample's bank range is divided into k equal-width intervals. Equal bank
values remain together and a constant bank occupies one bin. This replaces
the restore bundle's rank binning, which split ties and imposed near-uniform
bank-bin populations. For time-comparable absolute bank categories, specify
k−1 fixed internal thresholds, e.g. `"bankEdges": [100, 1000]` for k=3.
Outer bins include values outside those thresholds; boundary values enter
the bin on the right. Adaptive bins describe relative bank position, so a
bin transition need not imply crossing a fixed bank value.

Given the strategy and bank marginal probabilities, the requested outer
product is P⊗[s,b]=P(s)P(b), with 9 entries for k=3. We compute
H=−Σ p log₂(p), omitting zero entries. `product_bits` measures this distribution;
`product_normalized` divides it by log₂(3k).

We also report the empirical joint J[s,b]=N⁻¹Σᵢ 1(sᵢ=s)1(bᵢ=b), which is the
average of the **per-agent one-hot outer products**. `joint_bits` and
`joint_normalized` describe J. These are different operations:
H(P⊗)=H(S)+H(B), while H(J) retains strategy–bank dependence. Their difference
is `strategy_bank_mi_bits`. Both tables sum to one; the normalized entropies
lie in [0,1]. A finite sample may not attain the theoretical maximum.
These replace the active UI's distance/block-entropy plots. Legacy spatial
measurement functions remain available in `measurements.py`.

`--temporal` additionally counts observed same-agent (z at t−1, z at t)
pairs at each sampled t, regardless of `--sample-every`. The time-product
baseline is P(z at t−1)⊗P(z at t), with entropy H(z at t−1)+H(z at t).
The sparse observed joint retains temporal dependence. Both normalized
values divide by log₂((3k)²). We also report H(next|current)=H(pair)−H(current)
and I(current;next). At t=0 temporal values are NaN, not artificial zeros.
Transition IDs decode as `previous_id * (3*k) + next_id`; each state ID is
`strategy*k + bank_bin`. Unobserved cells have probability zero.

These are pooled empirical one-step measurements, not enumeration of every
possible next global configuration, an asymptotic entropy rate, or a claim
that the coarse state is Markov. Full conditional prediction would require
exposing the model's per-agent choice probabilities. No dense exponentially
large trajectory tensor is allocated.

## Verification

```sh
python -m unittest discover -s tests -v
```

The entropy tests check uniform, constant, correlated, independent and
identity-transition cases. Orchestration tests use NumPy/SciPy as a clearly
labelled CuPy stand-in to exercise the actual simulator, output sampling,
lag-one semantics, failures, interruption and CPU plotting. The separate
CuPy entropy parity test skips when CUDA is unavailable. A real GPU/Slurm
smoke run is still needed before a large cluster campaign.
