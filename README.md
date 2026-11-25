# SEA HMM processes

Install with `python -m pip install -e .`. For CUDA, use a CUDA-enabled PyTorch build. The models use CUDA when available and accept an explicit `device` such as `"cpu"` or `"cuda:0"`.

```python
from sea_hmm.hmm import CpGIslandModel, GaussianRegimeSwitchingHMM, Mess3

model = Mess3()
batch = model.sample([128, 64, 0])
filtered = model.filter(batch.observations, lengths=batch.lengths)
smoothed = model.smooth(batch.observations, lengths=batch.lengths)
exact = batch.hidden_states
```

For a row with `L` observations, the valid hidden state positions are `0` through `L`. Filtering at `t` uses `observations[:t]`. Smoothing uses `observations[:L]` at every state position. Inference always returns a batch dimension, with shape `[batch, observation_width + 1, num_states]`. Use `batch.lengths` to identify valid positions. Padding is zero in observations and posterior probabilities, and `-1` in exact hidden paths.

Mess3 uses symbol IDs `0, 1, 2`. The CpG Island Model uses IDs `0, 1, 2, 3` for `A, C, G, T`, with hidden states `0–3` for background and `4–7` for island, in the same base order. Both categorical models emit from the destination state, so `observations[t]` depends on `hidden_states[t + 1]` given the path. The Gaussian model emits scalar values from `hidden_states[t]`.

An `initial_probs` vector overrides the stationary start. For example, `Mess3(initial_probs=[1, 0, 0])` starts in state zero. To reproduce a batch, pass a seeded generator on the model device, such as `torch.Generator(device=model.device).manual_seed(42)`, to `sample`.

Run the tests with `python -m pip install -e '.[test]'` and `python -m pytest`.
