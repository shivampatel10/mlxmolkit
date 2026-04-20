import inspect

import pytest

from mlxmolkit.pipeline.driver import _RoundRobinRetryScheduler


def _dispatch_with_coalescing(
    scheduler: _RoundRobinRetryScheduler,
    batch_size: int,
    min_batch_size: int,
    lookahead_rounds: int,
) -> list[int]:
    """Call the optional coalescing API once it exists."""
    signature = inspect.signature(scheduler.dispatch)
    params = signature.parameters

    kwargs = {}
    for name in ("min_batch_size", "min_retry_batch_size", "retry_min_batch_size"):
        if name in params:
            kwargs[name] = min_batch_size
            break
    else:
        pytest.skip("retry coalescing min-batch parameter is not implemented")

    for name in ("lookahead_rounds", "max_lookahead_rounds", "lookahead"):
        if name in params:
            kwargs[name] = lookahead_rounds
            break
    else:
        pytest.skip("retry coalescing lookahead parameter is not implemented")

    return scheduler.dispatch(batch_size, **kwargs)


def test_retry_scheduler_advances_round_when_last_molecule_completed():
    scheduler = _RoundRobinRetryScheduler(n_mols=2, confs_per_mol=2, max_iterations=3)

    first_wave = scheduler.dispatch(4)
    assert first_wave == [0, 0, 1, 1]
    scheduler.record(first_wave, [False, False, True, True])

    second_wave = scheduler.dispatch(4)
    assert second_wave == [0, 0]
    scheduler.record(second_wave, [False, False])

    third_wave = scheduler.dispatch(4)
    assert third_wave == [0, 0]
    scheduler.record(third_wave, [False, False])

    assert scheduler.total_attempts == [6, 2]
    assert scheduler.dispatch(4) == []


def test_retry_scheduler_coalesces_future_attempts_without_exceeding_budget():
    scheduler = _RoundRobinRetryScheduler(n_mols=2, confs_per_mol=2, max_iterations=3)

    first_wave = scheduler.dispatch(4)
    assert first_wave == [0, 0, 1, 1]
    scheduler.record(first_wave, [False, False, True, True])

    retry_wave = _dispatch_with_coalescing(
        scheduler,
        batch_size=4,
        min_batch_size=4,
        lookahead_rounds=2,
    )

    assert retry_wave == [0, 0, 0, 0]
    assert scheduler.total_attempts == [6, 2]


def test_retry_scheduler_coalescing_skips_completed_molecules():
    scheduler = _RoundRobinRetryScheduler(n_mols=3, confs_per_mol=2, max_iterations=3)

    first_wave = scheduler.dispatch(6)
    assert first_wave == [0, 0, 1, 1, 2, 2]
    scheduler.record(first_wave, [False, False, True, True, False, False])

    retry_wave = _dispatch_with_coalescing(
        scheduler,
        batch_size=6,
        min_batch_size=6,
        lookahead_rounds=2,
    )

    assert 1 not in retry_wave
    assert retry_wave == [0, 0, 2, 2, 0, 0]
    assert scheduler.total_attempts == [6, 2, 4]


def test_retry_scheduler_coalescing_eventually_exhausts_budgets():
    scheduler = _RoundRobinRetryScheduler(n_mols=2, confs_per_mol=2, max_iterations=3)

    first_wave = scheduler.dispatch(4)
    assert first_wave == [0, 0, 1, 1]
    scheduler.record(first_wave, [False, False, True, True])

    retry_wave = _dispatch_with_coalescing(
        scheduler,
        batch_size=4,
        min_batch_size=4,
        lookahead_rounds=2,
    )
    scheduler.record(retry_wave, [False] * len(retry_wave))

    assert scheduler.total_attempts == [6, 2]
    assert scheduler.dispatch(4) == []
