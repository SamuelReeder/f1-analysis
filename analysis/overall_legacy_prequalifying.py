"""Legacy pre-qualifying development kernel under an explicit grid scenario.

Integrates a specified modular product of fits, not the successor joint model.
Official-grid policy, accepted fits and numerical forecast precision remain open.
"""
import hashlib
from pathlib import Path

import numpy as np

from analysis.overall_baselines import coefficient_draws, ordered_arrays
from analysis.overall_forecast_states import DEFAULT_DRAW_BATCH, _checked_parameter_digest
from analysis.overall_legacy_forecast import forecast_legacy_features
from analysis.overall_observation_kernels import (ObservationNoise, Q_PARAMETERS,
    order_log_probability, qualifying_segments)
from f1rank.overall.fitting import arrays_digest

POLICY = "legacy_prequalifying_cut_v1"
GRID_POLICY = "qualifying_order_no_penalties"
DEFINITION = Path(__file__).resolve().parents[1]/"docs/overall_legacy_prequalifying.md"


class LegacyObservationNoise(ObservationNoise):
    def draw(self, site, identity, **kwargs):
        return super().draw("legacy_qualifying:" + site, identity, **kwargs)


def forecast_legacy_qualifying(design, snapshots, posterior, *, cutoff, fit_metadata,
                               seed, training_arrays_key, advancing, grid_policy,
                               draw_batch_size=DEFAULT_DRAW_BATCH):
    """One all-set-times qualifying trajectory per paired qualifying draw.

    The complete future medians remain fixed inputs to the race mapping. Segment
    traces are temporary; the returned grid inventory retains every chain/draw.
    """
    if grid_policy != GRID_POLICY:
        raise ValueError("An explicit supported conditional grid policy is required")
    if (not snapshots or any(s.get("information_set") != "prequalifying" for s in snapshots)
            or not isinstance(advancing, dict)
            or set(advancing) != {s["event"]["event_id"] for s in snapshots}):
        raise ValueError("Require pre-qualifying fields and every event's advancing schedule")
    if type(draw_batch_size) is not int or draw_batch_size < 1:
        raise ValueError("Draw batch size must be a positive integer")
    if any(name not in posterior for name in Q_PARAMETERS):
        raise ValueError("Missing legacy qualifying observation draws")
    shape = np.shape(posterior.get("sd_level"))
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Qualifying draws require grouped chain/draw axes")
    parameters = {name:np.asarray(posterior[name]) for name in Q_PARAMETERS}
    for name,value in parameters.items():
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid qualifying observation draws for {name}")
        if ((value <= 0).any() if name in ("sigma0", "nu") else (value < 0).any()):
            raise ValueError(f"Invalid qualifying observation draws for {name}")
    observation_key = _checked_parameter_digest(parameters,draw_batch_size)
    result = forecast_legacy_features(design,snapshots,posterior,cutoff=cutoff,
        fit_metadata=fit_metadata,seed=seed,training_arrays_key=training_arrays_key,
        draw_batch_size=draw_batch_size)
    for event in result["events"]:
        positions = np.empty(shape+(len(event["drivers"]),),dtype=int)
        for start in range(0,shape[1],draw_batch_size):
            stop = min(start+draw_batch_size,shape[1])
            batch = dict(event,states={name:value[:,start:stop] for name,value in event["states"].items()})
            part = {name:value[:,start:stop] for name,value in parameters.items()}
            noise = LegacyObservationNoise(seed,shape,(start,stop))
            generated = qualifying_segments(batch,part,noise,advancing=advancing[event["event_id"]],full_df=parameters["nu"])
            positions[:,start:stop] = generated["positions"]
        event["grid_draws"] = positions
        event["advancing"] = list(advancing[event["event_id"]])
    result.update(purpose="development_legacy_prequalifying",prequalifying_policy=POLICY,
        grid_policy=grid_policy,qualifying_observation_key=observation_key,
        qualifying_interpretation="qualifying_positions_conditional_on_all_entrants_setting_times",
        definition_sha256=hashlib.sha256(DEFINITION.read_bytes()).hexdigest(),
        prequalifying_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grid_draws_key=arrays_digest({e["event_id"]:dict(positions=e["grid_draws"]) for e in result["events"]}),
        future_features_key=arrays_digest({e["event_id"]:e["race_features"] for e in result["events"]}),
        forecast_precision_verified=False,official_grid_policy_complete=False)
    return result


def integrate_legacy_prequalifying(forecast, coefficient_posterior, running_positions, *,
                                    arm="legacy_grid_qualifying", grid_batch_size=DEFAULT_DRAW_BATCH,
                                    coefficient_batch_size=DEFAULT_DRAW_BATCH):
    """Single-event marginal log masses over ALL grid/coefficient combinations.

    Aggregating equal grids preserves their empirical weights. Each coefficient
    vector stays paired; fits are never zipped or resized to equal lengths.
    This integrates the empirical product, not its Monte Carlo uncertainty.
    """
    if (forecast.get("purpose") != "development_legacy_prequalifying"
            or forecast.get("prequalifying_policy") != POLICY or forecast.get("grid_policy") != GRID_POLICY
            or forecast.get("feature_policy") != "median_of_paired_persistent_states_without_future_transients"):
        raise ValueError("Expected a complete legacy pre-qualifying forecast")
    if any(type(n) is not int or n < 1 for n in (grid_batch_size,coefficient_batch_size)):
        raise ValueError("Product batch sizes must be positive integers")
    events = forecast.get("events")
    shape = forecast.get("posterior_shape")
    if (not isinstance(events,list) or not events or not isinstance(shape,list) or len(shape)!=2
            or any(type(n) is not int or n<2 for n in shape)):
        raise ValueError("Incomplete qualifying draw/event inventory")
    if not isinstance(running_positions,dict) or set(running_positions)!={e["event_id"] for e in events}:
        raise ValueError("Every forecast event needs its full declared response panel")
    parameters = coefficient_draws(coefficient_posterior,arm)
    coefficient_shape = parameters["c_grid"].shape
    beta = {name:value.reshape(-1) for name,value in parameters.items()}
    n_beta = len(beta["c_grid"])
    rows, grids = [], []
    for event in events:
        field, panel = event["drivers"],running_positions[event["event_id"]]
        if event["catalog"]["information_set"] != "prequalifying":
            raise ValueError("Pre-qualifying integration cannot insert an actual-grid origin")
        if not isinstance(panel,dict) or set(panel)!=set(field):
            raise ValueError("Response panel must retain every declared driver")
        grid = np.asarray(event.get("grid_draws"))
        if (grid.shape!=tuple(shape)+(len(field),) or grid.dtype.kind not in "iu"
                or not np.array_equal(np.sort(grid,axis=-1),np.broadcast_to(np.arange(1,len(field)+1),grid.shape))):
            raise ValueError("Every qualifying draw needs a complete grid permutation")
        grids.append(grid)
        rows.append(dict(event_id=event["event_id"],drivers=field,grid_slot=np.arange(1,len(field)+1),
            running_position=[panel[d] for d in field],**event["race_features"]))
    # Validate the whole inventory before accumulating anything. Dummy slots
    # serve only the common panel validator; strengths below use every grid draw.
    checked = ordered_arrays(rows,arm=arm)
    key = arrays_digest({e["event_id"]:dict(positions=g) for e,g in zip(events,grids)})
    if key != forecast.get("grid_draws_key"):
        raise ValueError("Qualifying grid inventory changed after generation")
    feature_key = arrays_digest({e["event_id"]:e["race_features"] for e in events})
    if feature_key != forecast.get("future_features_key"):
        raise ValueError("Fixed qualifying features changed after generation")
    probabilities, unique_counts = [], []
    for event,row,grid in zip(events,rows,grids):
        unique,counts = np.unique(grid.reshape(-1,len(row["drivers"])),axis=0,return_counts=True)
        n_grid = int(np.prod(shape)); unique_counts.append(len(unique))
        if (np.asarray(row["running_position"]) > 0).sum() < 2:
            probabilities.append(0.)
            continue
        total = -np.inf
        for first in range(0,len(unique),grid_batch_size):
            last = min(first+grid_batch_size,len(unique))
            log_grid = np.log(unique[first:last])
            log_weight = np.log(counts[first:last])-np.log(n_grid)
            for start in range(0,n_beta,coefficient_batch_size):
                stop = min(start+coefficient_batch_size,n_beta)
                with np.errstate(over="ignore",invalid="ignore"):
                    strength = -beta["c_grid"][None,start:stop,None]*log_grid[:,None,:]
                    if arm=="legacy_grid_qualifying":
                        strength = (strength + beta["a_car"][None,start:stop,None]*row["car"]
                                    + beta["b_driver"][None,start:stop,None]*row["driver"])
                log_probability = order_log_probability(strength,np.asarray(row["running_position"]))
                weighted = log_probability+log_weight[:,None]-np.log(n_beta)
                total = np.logaddexp(total,np.logaddexp.reduce(weighted.reshape(-1)))
        probabilities.append(total)
    return dict(purpose="development_legacy_prequalifying_score",arm=arm,policy=POLICY,
        grid_policy=GRID_POLICY,event_ids=checked["event_ids"],fields=checked["fields"],
        log_probabilities=np.asarray(probabilities),axes=["event"],
        integration="full_empirical_product",unique_grid_counts=unique_counts,
        qualifying_posterior_shape=shape,coefficient_posterior_shape=list(coefficient_shape),
        qualifying_parameter_key=forecast["consumed_posterior_key"],
        qualifying_observation_key=forecast["qualifying_observation_key"],grid_draws_key=key,
        future_features_key=feature_key,
        coefficient_parameter_key=arrays_digest(dict(coefficients=parameters)),
        origins=[dict(event_id=e["event_id"],origin_at=e["catalog"]["origin_at"],
                      snapshot_sha256=e["catalog"]["snapshot_sha256"]) for e in events],
        source_availability_verified=False,convergence_verified=False,
        forecast_precision_verified=False,official_grid_policy_complete=False,
        independent_evidence_established=False,publication_allowed=False)
