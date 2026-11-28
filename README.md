# SEA HMM experiments

## Environment

From the repository root, run `uv sync --locked --extra test` to create the local `.venv` with the Python version in `.python-version` and the dependencies in `uv.lock`. Use `uv run --locked --extra test pytest` to run the tests and `uv run --locked python` to try the examples.

Sampling and inference use CUDA when the installed PyTorch build and the host driver support it. Set `device="cpu"` to run on CPU.

## State-emitting HMMs

```python
from sea_hmm.config import load_hmm

model = load_hmm("cfg/process/mess3.yaml")
batch = model.sample([128, 64, 0])
filtered = model.filter(batch.observations, lengths=batch.lengths)
smoothed = model.smooth(batch.observations, lengths=batch.lengths)
log_likelihood = model.log_likelihood(batch.observations, lengths=batch.lengths)
```

Each trajectory has one sampled hidden state per observation. At position `i`, filtering conditions on observations through `i` and smoothing conditions on the entire valid sequence. Posteriors have shape `[batch, observation_width, num_states]`. Observations and posteriors use zero padding and hidden paths use `-1`. Empty sequences have log likelihood zero.

### Process YAML

```yaml
type: discrete
name: Two-state example
num_states: 2
transition_probs:
  - [0.9, 0.1]
  - [0.2, 0.8]
emissions:
  - [0.7, 0.3]
  - [0.1, 0.9]
```

The optional `name` is a display label available as `model.name`. The type and matrices define the process.

- `transition_probs[i][j]` is the probability of moving from state `i` to state `j`. The matrix must have `num_states` rows and columns, and each row must sum to one.
- For `type: discrete`, `emissions[i][a]` is the probability that state `i` emits symbol ID `a`. Use one row per state and one column per symbol, with each row summing to one.
- For `type: continuous`, each emission row is `[mean, standard_deviation]` for a scalar Gaussian. Standard deviations must be positive.
- Set `initial_probs` to choose the initial state distribution. If omitted, the model uses the unique stationary distribution.

The examples are [Mess3](cfg/process/mess3.yaml), [CpG Island Model](cfg/process/cpg_island.yaml), and [Gaussian Regime-Switching HMM](cfg/process/gaussian_regime_switching.yaml). Mess3 emits IDs `0–2`. CpG emits IDs `0–3` for `A, C, G, T`, with states `0–3` for background and `4–7` for island in the same base order.

## Latent intent and external targets

```python
from sea_hmm.config import load_experiment

anchored = load_experiment("cfg/anchored_mess3.yaml")
pair = anchored.sample([8, 12], k=4)
observations = pair.trajectory.observations
short = anchored.predict_target(observations, lengths=pair.prefix_lengths, anchors=pair.anchors)
long = anchored.predict_target(observations, lengths=pair.trajectory.lengths, anchors=pair.anchors)
targets = pair.targets

static = load_experiment("cfg/static_mess3.yaml")
pair = static.sample([8, 12], k=4)
observations = pair.trajectory.observations
regime_probs = static.intent_posterior(observations, lengths=pair.prefix_lengths)
suffixes = static.sample_continuations(observations, k=4, lengths=pair.prefix_lengths, num_samples=16)
```

In the [anchored example](cfg/anchored_mess3.yaml), the intent is the hidden state at the end of the short prefix. The stored zero-based anchor selects that same state for both views. The anchor can also be configured as a fixed integer. In the [static example](cfg/static_mess3.yaml), the intent is the HMM component chosen for the whole episode. Its hidden path IDs are local to that component. Both constructions sample one target per episode from the configured `target_probs` row for that intent.

Experiment HMMs can be specified inline or loaded from a file relative to the experiment YAML. An anchored experiment needs one `target_probs` row per state. A static experiment needs one row per component, and all components must share the observation space. Loaders accept `device` and `dtype` arguments. To reproduce samples, pass a `torch.Generator(device=model.device).manual_seed(42)` to the sampling methods for that device.

Conditional sampling returns suffixes shaped `[batch, num_samples, k]`. For short categorical suffixes, compute exact conditional probabilities by subtracting the prefix log likelihood from the extended sequence log likelihood and exponentiating.

## Run a SEA experiment

```sh
uv run --locked python -m sea_hmm.experiments generate cfg/experiments/static_mess3_transformer.yaml
uv run --locked python -m sea_hmm.experiments run cfg/experiments/static_mess3_transformer.yaml
uv run --locked python -m sea_hmm.experiments run cfg/experiments/anchored_gaussian_lstm.yaml
```

The [static recipe](cfg/experiments/static_mess3_transformer.yaml) trains a four-layer, one-head Transformer on two Mess3 components with three target labels. The [anchored recipe](cfg/experiments/anchored_gaussian_lstm.yaml) trains an LSTM on scalar observations with a feature marking the anchor position. Both train on short and extended views of each episode with the same sampled target.

Set `data.mode` to `pre_generated` and run `generate` once to reuse a dataset. Set it to `online` for fresh training episodes each epoch and fixed validation and test splits. The YAML's `prefix_lengths` are sampled uniformly per episode and `lookahead` determines the additional length. Paths are relative to the YAML files that declare them. Set `device: cpu` for CPU execution, or leave `device: auto` to use CUDA when available.

`evaluation.modes` selects `oracle`, `ensemble`, or `single`. The oracle reports full specification uncertainty and exact classifier excess risk at the prefix. The ensemble and single-model estimators report a partial-clarification estimate and use HMM-conditional extensions. Ensemble epistemic disagreement is zero in single-model mode. The static recipe enumerates all 27 suffixes of length three. The Gaussian recipe draws 64 suffixes per prefix, so the oracle lookahead gain and gap carry Monte Carlo error. The estimated gain can exceed full specification uncertainty on individual prefixes with finite samples.

Each run writes `resolved.yaml`, `training_history.json`, `checkpoints/`, `metrics.json`, and `per_episode.csv` to `output_dir`. The metrics use natural logarithms and include short and full classifier accuracy and negative log likelihood, alongside the selected SEA estimators.
