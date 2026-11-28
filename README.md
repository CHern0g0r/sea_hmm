# SEA HMM experiments

## Environment

From the repository root, run `uv sync --locked --extra test` to create the local `.venv` with the Python version in `.python-version` and the dependencies in `uv.lock`. Use `uv run --locked --extra test pytest` to run the tests and `uv run --locked python` to try the examples.

Sampling and inference use CUDA when the installed PyTorch build and the host driver support it. Set `device="cpu"` to run on CPU.

## State-emitting HMMs

```python
from sea_hmm.config import load_hmm

model = load_hmm("examples/mess3.yaml")
batch = model.sample([128, 64, 0])
filtered = model.filter(batch.observations, lengths=batch.lengths)
smoothed = model.smooth(batch.observations, lengths=batch.lengths)
log_likelihood = model.log_likelihood(batch.observations, lengths=batch.lengths)
```

Each trajectory has one sampled hidden state per observation. At position `i`, filtering conditions on observations through `i` and smoothing conditions on the entire valid sequence. Posteriors have shape `[batch, observation_width, num_states]`. Observations and posteriors use zero padding and hidden paths use `-1`. Empty sequences have log likelihood zero.

The process YAML examples are [Mess3](examples/mess3.yaml), [CpG Island Model](examples/cpg_island.yaml), and [Gaussian Regime-Switching HMM](examples/gaussian_regime_switching.yaml). Load each with `load_hmm`. Their `params` configure the process and can include an `initial_probs` vector for the first emitting state. Mess3 emits IDs `0–2`. CpG emits IDs `0–3` for `A, C, G, T`, with states `0–3` for background and `4–7` for island in the same base order. The Gaussian process emits continuous scalars.

## Latent intent and external targets

```python
from sea_hmm.config import load_experiment

anchored = load_experiment("examples/anchored_mess3.yaml")
pair = anchored.sample([8, 12], k=4)
observations = pair.trajectory.observations
short = anchored.predict_target(observations, lengths=pair.prefix_lengths, anchors=pair.anchors)
long = anchored.predict_target(observations, lengths=pair.trajectory.lengths, anchors=pair.anchors)
targets = pair.targets

static = load_experiment("examples/static_mess3.yaml")
pair = static.sample([8, 12], k=4)
observations = pair.trajectory.observations
regime_probs = static.intent_posterior(observations, lengths=pair.prefix_lengths)
suffixes = static.sample_continuations(observations, k=4, lengths=pair.prefix_lengths, num_samples=16)
```

In the [anchored example](examples/anchored_mess3.yaml), the intent is the hidden state at the end of the short prefix. The stored zero-based anchor selects that same state for both views. The anchor can also be configured as a fixed integer. In the [static example](examples/static_mess3.yaml), the intent is the HMM component chosen for the whole episode. Its hidden path IDs are local to that component. Both constructions sample one target per episode from the configured `target_probs` row for that intent. Provide the anchor as input metadata to a classifier when it varies by example.

Experiment HMMs can be specified inline or loaded from a file relative to the experiment YAML. Loaders accept `device` and `dtype` arguments. To reproduce samples, pass a `torch.Generator(device=model.device).manual_seed(42)` to the sampling methods for that device.

Conditional sampling returns suffixes shaped `[batch, num_samples, k]`. For short categorical suffixes, compute exact conditional probabilities by subtracting the prefix log likelihood from the extended sequence log likelihood and exponentiating.
