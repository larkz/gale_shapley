# P2ETG and PrefLID

A two-sided online learning algorithm for stable matching, where agents learn their own preferences through pairwise comparison signals supplied by any external source.

# Quick Start

```bash
pip install pandas numpy matplotlib seaborn
python experiment.py
```

Produces:

  - `runs/` — per-run `rounds.csv`, `summary.json`, `config.json`, and aggregate `all_rounds.csv`, `all_summaries.csv`
  - `plots/` — `regret_curve.png`, `tstop_distribution.png`, `ci_convergence.png`, `regret_by_N.png`

# Directory Layout

```
project/
|-- gs_lib/
|   |-- gs_tools.py       # Man, Woman, Matching, GaleShapley, StabilityVerifier
|    -- bt.py             # Bradley-Terry MLE, CIs, ranking
|-- p2etg.py              # P2ETG learner + SignalProvider ABC
|-- providers.py          # BradleyTerryProvider (built-in)
|-- simulate.py           # Single-run demo
|-- experiment.py         # Multi-seed sweep + plotting
 -- runs/                 # Generated output
```

# The Interface

P2ETG is decoupled from the observation model. The environment supplies a `SignalProvider` — an object with a single method:

```python
def observe(self, agent, b1, b2) -> int:
    """Return 1 if agent prefers canonical-first of (b1, b2), else 0."""
```

**Canonical order:** $(b1, b2)$ if `hash(b1) <= hash(b2)`, else $(b2, b1)$.

Everything else — sampling schedule, MLE, confidence intervals, Gale--Shapley, stopping rule — is handled by P2ETG.

# The Two Entry Points

## Pull model — P2ETG asks the provider

```python
from p2etg import P2ETG
from providers import BradleyTerryProvider
import random

provider = BradleyTerryProvider(theta_men, theta_women, rng=random.Random(0))
learner = P2ETG(men, women, provider=provider, rng=random.Random(0))
result = learner.run_until_stop(adaptive=True, check_every=25, max_samples=200_000)

print(result["stopped"], result["T_stop"], result["matching"])
```

## Push model — the environment injects observations

```python
from p2etg import P2ETG, CallableProvider

# Null provider; we never call it.
learner = P2ETG(men, women, provider=CallableProvider(lambda *_: 0))

for t in range(1_000_000):
    agent = scheduler.pick_agent()
    b1, b2 = scheduler.pick_pair(agent)
    x = live_market.query(agent, b1, b2)
    learner.observe(agent, b1, b2, x)

    if t % 1000 == 0:
        learner._refresh_estimates()
        if learner._pairwise_disjoint():
            break

matching = learner.current_matching()
```

# Writing a Custom Provider

## As a class

```python
from p2etg import SignalProvider
from gs_lib.bt import _canonical

class MyProvider(SignalProvider):
    def observe(self, agent, b1, b2) -> int:
        key = _canonical(b1, b2)
        # Your logic here
        return 1 if my_model(agent, key[0]) > my_model(agent, key[1]) else 0
```

## As a function

```python
from p2etg import CallableProvider

def my_signal(agent, b1, b2) -> int:
    return 1 if score(agent, b1) > score(agent, b2) else 0

provider = CallableProvider(my_signal)
```

## Optional hooks

```python
class MyProvider(SignalProvider):
    def warmup(self, agent, partners):
        """Called once per agent before sampling begins."""
        self._cache[agent] = precompute(agent, partners)

    def observe(self, agent, b1, b2) -> int:
        ...

    def on_stop(self, committed_matching):
        """Called when P2ETG commits."""
        log_final_result(committed_matching)
```

# Provider Recipes

## Bradley--Terry (built-in)

```python
from providers import BradleyTerryProvider
provider = BradleyTerryProvider(theta_men, theta_women, rng=random.Random(0))
```

Bernoulli signal with $P(b_1  succ b_2) =  theta[b_1] / ( theta[b_1] +  theta[b_2])$.

## Noisy deterministic

Fixed rankings, flipped with probability `noise`:

```python
class NoisyProvider(SignalProvider):
    def __init__(self, rankings, noise=0.1, rng=None):
        self._rank = {a: {p: i for i, p in enumerate(prefs)}
                      for a, prefs in rankings.items()}
        self.noise = noise
        self.rng = rng or random.Random(0)

    def observe(self, agent, b1, b2) -> int:
        key = _canonical(b1, b2)
        r = self._rank[agent]
        true = 1 if r[key[0]] < r[key[1]] else 0
        if self.rng.random() < self.noise:
            return 1 - true
        return true
```

## Plackett--Luce

Softmax over partner scores:

```python
class PLProvider(SignalProvider):
    def __init__(self, scores, rng=None):
        self.scores = scores
        self.rng = rng or random.Random(0)

    def observe(self, agent, b1, b2) -> int:
        s = self.scores[agent]
        e1, e2 = math.exp(s[b1]), math.exp(s[b2])
        p = e1 / (e1 + e2)
        return 1 if self.rng.random() < p else 0
```

## Language model

Cache by canonical pair so each unique comparison is queried once:

```python
class LLMProvider(SignalProvider):
    def __init__(self, client, prompt_template):
        self.client = client
        self.prompt = prompt_template
        self._cache = {}

    def observe(self, agent, b1, b2) -> int:
        key = _canonical(b1, b2)
        if (agent, key) in self._cache:
            return self._cache[(agent, key)]
        prompt = self.prompt.format(agent=agent, a=key[0], b=key[1])
        resp = self.client.query(prompt).strip().upper()
        x = 1 if resp.startswith("A") else 0
        self._cache[(agent, key)] = x
        return x
```


# Running P2ETG

## `run_until_stop`

Blocks until the stopping condition fires or `max_samples` is hit.

```python
result = learner.run_until_stop(
    adaptive=True,       # True: fixed batches; False: doubling epochs
    check_every=25,      # samples per check (adaptive mode)
    max_epochs=400,      # doubling epochs cap (adaptive=False only)
    max_samples=100_000, # hard sample cap
    verbose=False,       # print a live trace per check
)
```

## `run_with_trace`

Same, plus returns a `rounds` list of `(t, matching, disjoint)` at each check.

```python
result = learner.run_with_trace(...)
for (t, matching, disjoint) in result["rounds"]:
    ...
```

## Manual stepping

```python
learner.step(100)            # draw 100 samples from least-sampled arms
learner._refresh_estimates() # recompute theta_hat, CIs, ranking
learner._pairwise_disjoint() # has the stopping condition fired?
```

# Result Object

`run_until_stop` and `run_with_trace` return a dict:

| } |  |  |
| — | — | — |
| **Key** | **Type** | **Meaning** |
| `matching` | Matching | Committed matching |
| `stopped` | bool | True if stopping condition fired |
| `T_stop` | int | Total samples drawn at commit |
| `epochs` | list[dict] | Per-check trace of diagnostics |
| `rounds` | list[tuple] | `(t, matching, disjoint)` per check (only `run_with_trace`) |

# Query API

```python
learner.is_stopped()                   # bool
learner.committed_matching()           # Matching | None
learner.current_matching()             # Matching (recomputed)
learner.estimated_preferences(agent)   # List[partner]
learner.state_snapshot()               # dict for logging
```

# Configuration Knobs

| l} |  |  |
| — | — | — |
| **Parameter** | **Meaning** | **Typical values** |
| `constant` | CI coefficient $w =  sqrt{c  log t / n}$ | 0.1 (prototype), 8.0 (conservative) |
| `check_every` | Samples per stopping check (adaptive) | 25 -- 500 |
| `max_epochs` | Doubling epochs cap (doubling mode) | 20 -- 400 |
| `max_samples` | Hard sample cap | $10^4$ -- $10^7$ |
| `adaptive` | Batch vs doubling schedule | True for prototyping |
| `rng` | `random.Random` for reproducibility | Shared with provider |

**Diagnostic signatures:**

| **Symptom** | **Diagnosis** | **Fix** |
| — | — | — |
| `stopped=False`, `correct=1` | CI too conservative | Lower `constant` |
| `stopped=True`, `correct=0` | CI fired early | Raise `constant` |

# Output Format for Plotting

Both `experiment.py` and any custom runner write the same shape:

```
runs/<tag>/
|-- rounds.csv     # columns: t, matching_str, disjoint, correct, regret, N, K, seed
|-- summary.json   # dict with stopped, T_stop, correct_at_stop, stability flags
 -- config.json    # the parameters used
```

`rounds.csv` gives one row per round $t = 1  ldots T_{stop} + 200$ (post-stop tail of 200).

Aggregate files at the top level:

```
runs/all_rounds.csv      # concatenation of every run's rounds
runs/all_summaries.csv   # one row per run
```


# End-to-End Custom Runner (Minimal)

```python
import json, random
from pathlib import Path
import numpy as np
import pandas as pd

from gs_lib.gs_tools import Man, Woman, PreferenceList, GaleShapley, StabilityVerifier
from p2etg import P2ETG, SignalProvider
from gs_lib.bt import _canonical

class MyProvider(SignalProvider):
    def observe(self, agent, b1, b2) -> int:
        key = _canonical(b1, b2)
        return int(my_score(agent, key[0]) > my_score(agent, key[1]))

def run(N, K, seed, out_dir: Path):
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    men   = [Man(f"m{i}")   for i in range(1, N + 1)]
    women = [Woman(f"w{j}") for j in range(1, K + 1)]

    # Ground truth for the oracle (optional).
    rankings = {**{m: sorted(women, key=lambda w: -np_rng.random()) for m in men},
                **{w: sorted(men,   key=lambda m: -np_rng.random()) for w in women}}
    h_star = GaleShapley(PreferenceList(rankings)).find_stable_matching("men")

    provider = MyProvider()
    learner = P2ETG(men, women, provider=provider, rng=rng, constant=0.1)
    result = learner.run_with_trace(adaptive=True, check_every=25,
                                    max_samples=100_000, verbose=False)

    committed = result["matching"]
    rows, prev_t = [], 0
    for (t, m, disj) in result["rounds"]:
        for tt in range(prev_t + 1, t + 1):
            rows.append({"t": tt, "matching_str": str(m),
                         "disjoint": disj, "correct": int(m == h_star)})
        prev_t = t
    if result["stopped"]:
        for tt in range(prev_t + 1, prev_t + 201):
            rows.append({"t": tt, "matching_str": str(committed),
                         "disjoint": True, "correct": int(committed == h_star)})

    cum = 0
    for r in rows:
        cum += 1 - r["correct"]
        r["regret"] = cum

    ok_hat, _, _ = StabilityVerifier(
        learner._build_preference_lists()).is_stable(committed)
    ok_true, _, _ = StabilityVerifier(
        PreferenceList(rankings)).is_stable(committed)

    summary = {
        "N": N, "K": K, "seed": seed,
        "stopped": result["stopped"], "T_stop": result["T_stop"],
        "correct_at_stop": int(committed == h_star),
        "final_regret": rows[-1]["regret"] if rows else None,
        "stable_under_truth": int(ok_true),
        "stable_under_hat": int(ok_hat),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "rounds.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    for seed in range(5):
        run(4, 4, seed, Path(f"runs_mine/N4_K4_seed{seed}"))
        print(f"seed {seed} done")
```

Then:

```python
df = pd.read_csv("runs_mine/N4_K4_seed0/rounds.csv")
print(df[df["disjoint"]].head(1))   # first disjoint=True is T_stop
```


# Summary

> 
 To use P2ETG, define one function: `observe(agent, b1, b2) -> int`. Wrap it in a `SignalProvider`, pass it to `P2ETG(men, women, provider)`, and call `run_until_stop()`. The result is a committed stable matching, and the environment's only job is to produce binary comparison signals.}

The algorithm is fully decoupled from the signal source. Swap BT for a language model, a human, a live market, or a composition — P2ETG runs unchanged.
