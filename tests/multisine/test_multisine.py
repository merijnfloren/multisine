from numbers import Real

import numpy as np
import pytest
from numpy.typing import ArrayLike

from multisine import (
    RandomPhaseMultisine,
    random_phase_multisine,
    random_phase_orthogonal_multisine,
)


def _assert_requested_rms_amplitude(
    multisine: RandomPhaseMultisine,
    amplitude: float | tuple[float, ...],
) -> None:
    if isinstance(amplitude, float):
        actual_amplitude = np.sqrt(np.mean(multisine.u**2))
    else:
        channel_axis = 1
        reduction_axes = (0, *range(channel_axis + 1, multisine.u.ndim))
        actual_amplitude = np.sqrt(np.mean(multisine.u**2, axis=reduction_axes))

    np.testing.assert_allclose(actual_amplitude, amplitude)


def test_random_phase_multisine_has_requested_scalar_rms_amplitude() -> None:
    amplitude = 1.5

    multisine = random_phase_multisine(
        n_samples=64,
        fs=100.0,
        amplitude=amplitude,
        nu=2,
        n_realizations=3,
    )

    assert multisine.u.shape == (64, 2, 3)
    _assert_requested_rms_amplitude(multisine, amplitude)


def test_random_phase_multisine_omits_singleton_realization_axis() -> None:
    multisine = random_phase_multisine(n_samples=64, fs=100.0, nu=2)

    assert multisine.u.shape == (64, 2)


def test_random_phase_multisine_has_requested_per_channel_rms_amplitudes() -> None:
    amplitude = (1.0, 2.0)

    multisine = random_phase_multisine(
        n_samples=64,
        fs=100.0,
        amplitude=amplitude,
        nu=2,
        n_realizations=3,
    )

    _assert_requested_rms_amplitude(multisine, amplitude)


@pytest.mark.parametrize(
    ("amplitude", "expected_amplitude"),
    [
        (1, 1.0),
        (np.float32(1.5), 1.5),
        ((1, 2), (1.0, 2.0)),
        ([1, 2], (1.0, 2.0)),
        (range(1, 3), (1.0, 2.0)),
        (np.array([1, 2]), (1.0, 2.0)),
    ],
)
def test_random_phase_multisine_normalizes_valid_amplitudes(
    amplitude: Real | ArrayLike,
    expected_amplitude: float | tuple[float, ...],
) -> None:
    multisine = random_phase_multisine(
        n_samples=64,
        fs=100.0,
        amplitude=amplitude,
        nu=2,
        n_realizations=3,
    )

    assert multisine.amplitude == expected_amplitude
    _assert_requested_rms_amplitude(multisine, expected_amplitude)


@pytest.mark.parametrize(
    ("amplitude", "exception"),
    [
        (np.array([[1.0, 2.0]]), TypeError),
        ([True, 1.0], TypeError),
        (np.arange(2), ValueError),
        (range(2), ValueError),
    ],
)
def test_random_phase_multisine_rejects_invalid_amplitudes(
    amplitude: Real | ArrayLike,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        random_phase_multisine(n_samples=64, fs=100.0, amplitude=amplitude, nu=2)


def test_random_phase_multisine_is_reproducible_for_same_seed() -> None:
    kwargs = {
        "n_samples": 64,
        "fs": 100.0,
        "amplitude": (1.0, 2.0),
        "nu": 2,
        "n_realizations": 3,
        "seed": 13,
    }

    first = random_phase_multisine(**kwargs)
    second = random_phase_multisine(**kwargs)

    np.testing.assert_array_equal(first.u, second.u)


def test_random_phase_multisine_differs_for_different_seeds() -> None:
    kwargs = {
        "n_samples": 64,
        "fs": 100.0,
        "amplitude": (1.0, 2.0),
        "nu": 2,
        "n_realizations": 3,
    }

    first = random_phase_multisine(**kwargs, seed=13)
    second = random_phase_multisine(**kwargs, seed=14)

    assert not np.array_equal(first.u, second.u)


def test_random_phase_orthogonal_multisine_has_requested_scalar_rms_amplitude() -> None:
    amplitude = 1.5

    multisine = random_phase_orthogonal_multisine(
        n_samples=64,
        fs=100.0,
        nu=2,
        amplitude=amplitude,
        n_experiments=3,
    )

    assert multisine.u.shape == (64, 2, 2, 3)
    _assert_requested_rms_amplitude(multisine, amplitude)


def test_random_phase_orthogonal_multisine_omits_singleton_experiment_axis() -> None:
    multisine = random_phase_orthogonal_multisine(n_samples=64, fs=100.0, nu=2)

    assert multisine.u.shape == (64, 2, 2)


def test_random_phase_orthogonal_multisine_has_requested_per_channel_rms_amplitudes() -> None:
    amplitude = (1.0, 2.0)

    multisine = random_phase_orthogonal_multisine(
        n_samples=64,
        fs=100.0,
        nu=2,
        amplitude=amplitude,
        n_experiments=3,
    )

    _assert_requested_rms_amplitude(multisine, amplitude)


def test_random_phase_orthogonal_multisine_is_reproducible_for_same_seed() -> None:
    kwargs = {
        "n_samples": 64,
        "fs": 100.0,
        "nu": 2,
        "amplitude": (1.0, 2.0),
        "n_experiments": 3,
        "seed": 13,
    }

    first = random_phase_orthogonal_multisine(**kwargs)
    second = random_phase_orthogonal_multisine(**kwargs)

    np.testing.assert_array_equal(first.u, second.u)


def test_random_phase_orthogonal_multisine_differs_for_different_seeds() -> None:
    kwargs = {
        "n_samples": 64,
        "fs": 100.0,
        "nu": 2,
        "amplitude": (1.0, 2.0),
        "n_experiments": 3,
    }

    first = random_phase_orthogonal_multisine(**kwargs, seed=13)
    second = random_phase_orthogonal_multisine(**kwargs, seed=14)

    assert not np.array_equal(first.u, second.u)


def test_random_phase_orthogonal_multisine_is_unitary_at_excited_frequencies() -> None:
    multisine = random_phase_orthogonal_multisine(
        n_samples=64,
        fs=100.0,
        nu=3,
        n_experiments=2,
    )

    spectrum = np.fft.rfft(multisine.u, axis=0)
    spectrum = spectrum[multisine.freq.excited_bins]
    spectrum = np.moveaxis(spectrum, -1, 1)

    row_norms = np.linalg.norm(spectrum, axis=-1)
    scales = row_norms[..., 0]
    unitary_spectrum = spectrum / scales[..., None, None]

    inverse_spectrum = np.linalg.solve(unitary_spectrum, np.eye(unitary_spectrum.shape[-1]))
    conjugate_transpose = unitary_spectrum.conj().swapaxes(-1, -2)
    np.testing.assert_allclose(inverse_spectrum, conjugate_transpose)
