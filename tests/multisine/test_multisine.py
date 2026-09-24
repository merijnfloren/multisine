from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

from multisine import (
    RandomPhaseMultisine,
    random_phase_multisine,
    random_phase_orthogonal_multisine,
)
from multisine._multisine import _convert_amplitude, _validate_amplitude


@dataclass(frozen=True)
class GeneratorCase:
    """A public multisine generator and its expected output shapes."""

    name: str
    generator: Callable[..., RandomPhaseMultisine]
    count_keyword: Literal["n_realizations", "n_experiments"]
    repeated_shape: tuple[int, ...]
    singleton_shape: tuple[int, ...]


@pytest.fixture(
    params=[
        GeneratorCase(
            name="random phase",
            generator=random_phase_multisine,
            count_keyword="n_realizations",
            repeated_shape=(64, 2, 3),
            singleton_shape=(64, 2),
        ),
        GeneratorCase(
            name="orthogonal",
            generator=random_phase_orthogonal_multisine,
            count_keyword="n_experiments",
            repeated_shape=(64, 2, 2, 3),
            singleton_shape=(64, 2, 2),
        ),
    ],
    ids=lambda case: case.name,
)
def generator_case(request: pytest.FixtureRequest) -> GeneratorCase:
    assert isinstance(request.param, GeneratorCase)
    return request.param


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


def _generate(
    generator_case: GeneratorCase,
    *,
    amplitude: Real | ArrayLike = 1.0,
    count: int = 1,
    seed: int = 42,
) -> RandomPhaseMultisine:
    common_kwargs = {
        "n_samples": 64,
        "fs": 100.0,
        "nu": 2,
        "amplitude": amplitude,
        "seed": seed,
    }
    if generator_case.count_keyword == "n_realizations":
        return generator_case.generator(
            **common_kwargs,
            n_realizations=count,
        )
    return generator_case.generator(
        **common_kwargs,
        n_experiments=count,
    )


def test_generators_have_requested_scalar_rms_amplitude(
    generator_case: GeneratorCase,
) -> None:
    amplitude = 1.5

    multisine = _generate(generator_case, amplitude=amplitude, count=3)

    assert multisine.u.shape == generator_case.repeated_shape
    assert multisine.amplitude == amplitude
    _assert_requested_rms_amplitude(multisine, amplitude)


def test_generators_omit_singleton_count_axis(generator_case: GeneratorCase) -> None:
    multisine = _generate(generator_case)

    assert multisine.u.shape == generator_case.singleton_shape


def test_generators_have_requested_per_channel_rms_amplitudes(
    generator_case: GeneratorCase,
) -> None:
    amplitude = (1.0, 2.0)

    multisine = _generate(generator_case, amplitude=amplitude, count=3)

    assert multisine.amplitude == amplitude
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
def test_validate_amplitude_accepts_valid_values(
    amplitude: Real | ArrayLike,
    expected_amplitude: float | tuple[float, ...],
) -> None:
    _validate_amplitude(amplitude, nu=2)

    assert _convert_amplitude(amplitude) == expected_amplitude


@pytest.mark.parametrize(
    ("amplitude", "exception"),
    [
        (np.array([[1.0, 2.0]]), TypeError),
        ([True, 1.0], TypeError),
        (np.arange(2), ValueError),
        (range(2), ValueError),
    ],
)
def test_validate_amplitude_rejects_invalid_values(
    amplitude: Real | ArrayLike,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _validate_amplitude(amplitude, nu=2)


def test_generators_are_reproducible_for_same_seed(generator_case: GeneratorCase) -> None:
    first = _generate(generator_case, amplitude=(1.0, 2.0), count=3, seed=13)
    second = _generate(generator_case, amplitude=(1.0, 2.0), count=3, seed=13)

    np.testing.assert_array_equal(first.u, second.u)


def test_generators_differ_for_different_seeds(generator_case: GeneratorCase) -> None:
    first = _generate(generator_case, amplitude=(1.0, 2.0), count=3, seed=13)
    second = _generate(generator_case, amplitude=(1.0, 2.0), count=3, seed=14)

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
