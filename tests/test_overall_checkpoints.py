"""Checkpoint corruption, crash-boundary and numerical replay specifications.

These are engineering checks, not synthetic recovery or model acceptance.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from f1rank.overall.checkpoints import Checkpoints, decode_state, encode_state
from f1rank.overall.resumable_sampling import run_chunks


def identity():
    return {"settings": {"chains": 2, "warmup": 4, "samples": 6}, "target": "fixture"}


def state(draws=0):
    from numpyro.infer.hmc import HMCState
    from numpyro.infer.hmc_util import HMCAdaptState
    z = {"beta": np.zeros((2, 3), np.float32)}
    adapt = HMCAdaptState(
        np.ones(2, np.float32), {("beta",): np.ones((2, 3), np.float32)},
        {("beta",): np.ones((2, 3), np.float32)},
        {("beta",): np.ones((2, 3), np.float32)},
        (np.zeros(2, np.float32),), (np.ones(2, np.int32),),
        np.zeros(2, np.int32), np.array([[2, 3], [4, 5]], np.uint32))
    return HMCState(
        np.full(2, 4+draws, np.int32), z, copy.deepcopy(z),
        np.ones(2, np.float32), np.ones(2, np.float32), None, None,
        np.ones(2, np.int32), np.ones(2, np.float32), np.ones(2, np.float32),
        np.zeros(2, bool), adapt, np.array([[6, 7], [8, 9]], np.uint32))


def samples(draws):
    return ({"beta": np.arange(2*draws*3, dtype=np.float32).reshape(2, draws, 3)},
            {"diverging": np.zeros((2, draws), bool),
             "num_steps": np.ones((2, draws), np.int32),
             "accept_prob": np.full((2, draws), .8, np.float32)})


def test_round_trip_preserves_state_container_mass_blocks_and_random_keys():
    import jax
    import jax.numpy as jnp
    original = state()._replace(
        rng_key=jax.random.split(jax.random.key(19), 2),
        trajectory_length=jnp.broadcast_to(jnp.asarray(1.), (2,)))
    tree, leaves = encode_state(original)
    restored = decode_state(json.loads(json.dumps(tree)), leaves)
    assert type(restored) is type(original)
    assert type(restored.adapt_state) is type(original.adapt_state)
    assert list(restored.adapt_state.inverse_mass_matrix) == [("beta",)]
    assert restored.trajectory_length.weak_type == original.trajectory_length.weak_type
    np.testing.assert_array_equal(jax.random.key_data(restored.rng_key), jax.random.key_data(original.rng_key))
    np.testing.assert_array_equal(restored.z["beta"], original.z["beta"])
    assert restored.r is None


def test_warmup_then_chunks_round_trip_and_retains_every_draw(tmp_path):
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        posterior, extras = samples(4)
        saved.append(state(4), posterior=posterior, extras=extras)
    with Checkpoints(tmp_path, identity()) as saved:
        assert saved.draws == 4
        np.testing.assert_array_equal(saved.state().rng_key, state(4).rng_key)
        posterior, extras = samples(2)
        saved.append(state(6), posterior=posterior, extras=extras)
        rows = list(saved.chunks())
        assert [(r["start_draw"], r["draws"]) for r, _, _ in rows] == [(0, 4), (4, 6)]
        assert np.concatenate([p["beta"] for _, p, _ in rows], axis=1).shape == (2, 6, 3)


def test_same_target_identity_and_exclusive_lock_are_required(tmp_path):
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        with pytest.raises(BlockingIOError):
            with Checkpoints(tmp_path, identity()):
                pass
    changed = identity()
    changed["target"] = "new-race"
    with pytest.raises(ValueError, match="different target"):
        with Checkpoints(tmp_path, changed):
            pass


def test_kill_before_atomic_rename_keeps_previous_generation(tmp_path, monkeypatch):
    original = Path.rename
    def fail_partial(self, target):
        if self.name.startswith(".partial-"):
            raise InterruptedError("simulated interruption before commit")
        return original(self, target)
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        with monkeypatch.context() as context:
            context.setattr(Path, "rename", fail_partial)
            posterior, extras = samples(2)
            with pytest.raises(InterruptedError):
                saved.append(state(2), posterior=posterior, extras=extras)
    assert list(tmp_path.glob(".partial-*"))
    with Checkpoints(tmp_path, identity()) as saved:
        assert saved.draws == 0
        posterior, extras = samples(2)
        saved.append(state(2), posterior=posterior, extras=extras)
        assert saved.draws == 2


@pytest.mark.parametrize("name", ["state.npz", "posterior.npz", "extras.npz"])
def test_corrupted_committed_bytes_fail_closed(tmp_path, name):
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        posterior, extras = samples(2)
        saved.append(state(2), posterior=posterior, extras=extras)
    path = tmp_path / "checkpoint-000001" / name
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="bytes changed"):
        with Checkpoints(tmp_path, identity()):
            pass


@pytest.mark.parametrize("failure", ["iteration", "overrun", "missing_site", "chains"])
def test_invalid_next_chunk_cannot_advance_checkpoint(tmp_path, failure):
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        posterior, extras = samples(2)
        saved.append(state(2), posterior=posterior, extras=extras)
        next_state = state(4)
        if failure == "iteration":
            next_state = state(5)
        elif failure == "overrun":
            posterior, extras = samples(5)
            next_state = state(7)
        elif failure == "missing_site":
            posterior = {"different_site": posterior["beta"]}
        elif failure == "chains":
            posterior = {k: v[:1] for k, v in posterior.items()}
            extras = {k: v[:1] for k, v in extras.items()}
        with pytest.raises(ValueError):
            saved.append(next_state, posterior=posterior, extras=extras)
        assert saved.draws == 2
        assert len(list(tmp_path.glob("checkpoint-*"))) == 2


def test_missing_generation_and_changed_manifest_chain_fail_closed(tmp_path):
    with Checkpoints(tmp_path, identity()) as saved:
        saved.append(state())
        posterior, extras = samples(2)
        saved.append(state(2), posterior=posterior, extras=extras)
    first = tmp_path / "checkpoint-000000"
    first.rename(tmp_path / ".removed")
    with pytest.raises(ValueError, match="contiguous"):
        with Checkpoints(tmp_path, identity()):
            pass
    (tmp_path / ".removed").rename(first)
    path = first / "manifest.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps(record, indent=2))
    with pytest.raises(ValueError, match="manifest chain"):
        with Checkpoints(tmp_path, identity()):
            pass


def small_model(data):
    import numpyro
    import numpyro.distributions as dist
    x = numpyro.sample("x", dist.Normal(0., 1.))
    numpyro.deterministic("twice_x", x*2)
    numpyro.sample("observations", dist.Normal(x, .7), obs=data)


def numerical_settings():
    from f1rank.overall.fitting import FIT_SETTINGS
    return dict(FIT_SETTINGS, chains=1, warmup=32, samples=25, max_tree_depth=5, seed=17)


def collected(directory):
    identity_value = json.loads((directory / "identity.json").read_text())
    with Checkpoints(directory, identity_value) as saved:
        rows = list(saved.chunks())
        post = {k: np.concatenate([p[k] for _, p, _ in rows], axis=1) for k in rows[0][1]}
        extras = {k: np.concatenate([e[k] for _, _, e in rows], axis=1) for k in rows[0][2]}
    return post, extras


@pytest.mark.parametrize("chains", [1, 2])
def test_actual_nuts_chunks_equal_uninterrupted_sampling(tmp_path, chains):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_median
    from f1rank.overall.inference import EXTRA_FIELDS
    settings = dict(numerical_settings(), chains=chains)
    data = jnp.array([-.3, .1, .6], dtype=jnp.float32)
    result = run_chunks(small_model, (data,), kwargs={}, settings=settings,
        run_key=hashlib.sha256(b"small-model-fixed-data").hexdigest(), directory=tmp_path,
        chunk_draws=7, verify_inputs=lambda: None)
    assert result["state"] == "draws_complete"
    assert result["publication_allowed"] is False
    sampler = MCMC(NUTS(small_model, target_accept_prob=settings["target_accept"],
                       max_tree_depth=settings["max_tree_depth"],
                       init_strategy=init_to_median(num_samples=settings["init_median_samples"])),
                   num_warmup=settings["warmup"], num_samples=settings["samples"],
                   num_chains=settings["chains"], chain_method="parallel", progress_bar=False)
    sampler.run(jax.random.PRNGKey(settings["seed"]), data, extra_fields=EXTRA_FIELDS)
    posterior, extras = collected(tmp_path)
    for key, value in sampler.get_samples(group_by_chain=True).items():
        np.testing.assert_array_equal(posterior[key], value)
    for key, value in sampler.get_extra_fields(group_by_chain=True).items():
        np.testing.assert_array_equal(extras[key], value)
    jax.clear_caches()


def test_actual_process_restart_after_committed_chunk_reproduces_every_sample(tmp_path):
    """Use separate interpreters so recovery cannot rely on a live sampler."""
    script = tmp_path / "sample.py"
    script.write_text('''
import hashlib, sys
from pathlib import Path
import jax.numpy as jnp
from f1rank.overall.checkpoints import Checkpoints
from f1rank.overall.resumable_sampling import run_chunks
from test_overall_checkpoints import small_model, numerical_settings
original = Checkpoints.append
def interrupted(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if result["draws"] == 7:
        raise SystemExit(23)
    return result
if sys.argv[2] == "interrupt":
    Checkpoints.append = interrupted
result = run_chunks(small_model, (jnp.array([-.3,.1,.6], dtype=jnp.float32),),
    kwargs={}, settings=numerical_settings(), chunk_draws=7,
    run_key=hashlib.sha256(b"small-model-fixed-data").hexdigest(),
    directory=Path(sys.argv[1]), verify_inputs=lambda: None)
assert result["state"] == "draws_complete"
''')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root/"tests"))))
    resumed, uninterrupted = tmp_path / "resumed", tmp_path / "uninterrupted"
    first = subprocess.run([sys.executable, str(script), str(resumed), "interrupt"],
                           env=env, capture_output=True, text=True, timeout=180)
    assert first.returncode == 23, first.stderr
    for directory in (resumed, uninterrupted):
        result = subprocess.run([sys.executable, str(script), str(directory), "complete"],
                                env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
    for actual, expected in zip(collected(resumed), collected(uninterrupted)):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_changed_inputs_before_commit_do_not_advance_sampling(tmp_path):
    import jax.numpy as jnp
    def check():
        if (tmp_path / "checkpoint-000000").exists():
            raise ValueError("source changed")
    with pytest.raises(ValueError, match="source changed"):
        run_chunks(small_model, (jnp.array([.2]),), kwargs={}, settings=numerical_settings(),
            chunk_draws=7, run_key="a"*64, directory=tmp_path, verify_inputs=check)
    ident = json.loads((tmp_path / "identity.json").read_text())
    with Checkpoints(tmp_path, ident) as saved:
        assert saved.draws == 0
        assert len(saved.records) == 1
