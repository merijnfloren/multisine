from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

MIN_NUMBER_OF_SAMPLES = 4  # keep in sync with docstrings


@dataclass
class FrequencyInfo:
    """Metadata for the frequency content of a multisine signal.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    f_res : float
        Frequency resolution in Hz.
    f_min : float
        Minimum frequency in Hz (snapped to grid).
    f_max : float
        Maximum frequency in Hz (snapped to grid).
    freqs : NDArray[np.floating[Any]]
        Non-negative frequencies in Hz, shape (n_samples // 2 + 1,).
    excited_bins : NDArray[np.int_]
        Indices of excited bins in freqs.
    non_excited_bins : NDArray[np.int_]
        Indices of non-excited bins in freqs.

    """

    fs: float
    f_res: float
    f_min: float
    f_max: float
    freqs: NDArray[np.floating[Any]]
    excited_bins: NDArray[np.int_]
    non_excited_bins: NDArray[np.int_]


@dataclass
class RandomPhaseMultisine:
    """A random-phase multisine signal with associated metadata.

    Attributes
    ----------
    u : NDArray[np.floating[Any]]
        The time-domain multisine signal. For a random-phase multisine, it has
        shape ``(n_samples, nu)`` when ``n_realizations`` is one, or
        ``(n_samples, nu, n_realizations)`` otherwise. For an orthogonal
        multisine, it has shape ``(n_samples, nu, nu)`` when ``n_experiments`` is
        one, or ``(n_samples, nu, nu, n_experiments)`` otherwise. Here, ``n_samples``
        is the number of samples and ``nu`` is the number of input channels.
    freq : FrequencyInfo
        Frequency metadata of the multisine signal.
    amplitude : float or tuple of float
        The requested time-domain root-mean-square (RMS) amplitude of the
        multisine signal. If a float, the time-domain RMS amplitude is the same
        for all channels and realizations. If a tuple, it contains one
        time-domain RMS amplitude per input channel.
    seed : int
        The random seed used to generate the multisine signal.

    """

    u: NDArray[np.floating[Any]]
    freq: FrequencyInfo
    amplitude: float | tuple[float, ...]
    seed: int


def random_phase_multisine(
    n_samples: int,
    fs: float,
    *,
    amplitude: float | tuple[float, ...] = 1.0,
    nu: int = 1,
    n_realizations: int = 1,
    f_min: float | None = None,
    f_max: float | None = None,
    seed: int = 42,
) -> RandomPhaseMultisine:
    """Generate a random-phase multisine signal.

    Parameters
    ----------
    n_samples : int
        Number of samples in each realization.
    fs : float
        Sampling frequency in Hz.
    amplitude : float or tuple of float, optional
        Desired time-domain root-mean-square (RMS) amplitude. A float sets the
        time-domain RMS amplitude over all channels and realizations. A tuple
        supplies one time-domain RMS amplitude per input channel and must have
        length ``nu``. Default is ``1.0``.
    nu : int, optional
        Number of input channels. Default is ``1``.
    n_realizations : int, optional
        Number of independent phase realizations per input channel. Default is ``1``.
    f_min : float, optional
        Lowest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the first non-DC frequency bin.
    f_max : float, optional
        Highest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the bin below the Nyquist frequency.
    seed : int, optional
        Seed for the random phase generator. Default is ``42``.

    Returns
    -------
    RandomPhaseMultisine
        The generated signal, with shape ``(n_samples, nu)`` when
        ``n_realizations`` is one and ``(n_samples, nu, n_realizations)``
        otherwise; its frequency information; the requested time-domain RMS
        amplitude; and the random seed used.

    """
    _validate_arguments(
        n_samples=n_samples,
        fs=fs,
        amplitude=amplitude,
        nu=nu,
        additional_count=n_realizations,
        additional_count_name="n_realizations",
        f_min=f_min,
        f_max=f_max,
        seed=seed,
    )

    freq = _create_frequency_info(n_samples, fs, f_min, f_max)
    rng = np.random.default_rng(seed)

    n_freqs = freq.freqs.size

    # Create multisine in frequency domain
    spectrum = np.zeros((n_freqs, nu, n_realizations), dtype=complex)
    spectrum[freq.excited_bins] = 1 + 0j

    phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_realizations))
    spectrum *= np.exp(phase)

    # Convert to time domain
    u = np.fft.irfft(spectrum, n=n_samples, axis=0)

    u = _ensure_requested_amplitude(u, amplitude)

    if n_realizations == 1:
        u = u[..., 0]

    return RandomPhaseMultisine(u=u, freq=freq, amplitude=amplitude, seed=seed)


def random_phase_orthogonal_multisine(
    n_samples: int,
    fs: float,
    nu: int,
    *,
    amplitude: float | tuple[float, ...] = 1.0,
    n_experiments: int = 1,
    independent_subexperiments: bool = True,
    f_min: float | None = None,
    f_max: float | None = None,
    seed: int = 42,
) -> RandomPhaseMultisine:
    """Generate random-phase orthogonal multisine signals.

    Parameters
    ----------
    n_samples : int
        Number of samples in each experiment.
    fs : float
        Sampling frequency in Hz.
    nu : int
        Number of input channels.
    amplitude : float or tuple of float, optional
        Desired time-domain root-mean-square (RMS) amplitude. A float sets the
        time-domain RMS amplitude over all channels and experiments. A tuple
        supplies one time-domain RMS amplitude per input channel and must have
        length ``nu``. Default is ``1.0``.
    n_experiments : int, optional
        Number of orthogonal multisine experiments. Default is ``1``.
    independent_subexperiments : bool, optional
        Whether to apply an independently drawn additional phase to each of the
        ``nu`` orthogonal subexperiments within every experiment. This allows
        estimation of the covariance of stochastic nonlinear output distortions
        without compromising orthogonality. Default is ``True``.
    f_min : float, optional
        Lowest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the first non-DC frequency bin.
    f_max : float, optional
        Highest excited frequency in Hz. The value is snapped to the frequency
        grid. Default is the bin below the Nyquist frequency.
    seed : int, optional
        Seed for the random phase generator. Default is ``42``.

    Returns
    -------
    RandomPhaseMultisine
        The generated signal, with shape ``(n_samples, nu, nu)`` when
        ``n_experiments`` is one and ``(n_samples, nu, nu, n_experiments)``
        otherwise; its frequency information; the requested time-domain RMS
        amplitude; and the random seed used.

    """
    _validate_arguments(
        n_samples=n_samples,
        fs=fs,
        amplitude=amplitude,
        nu=nu,
        additional_count=n_experiments,
        additional_count_name="n_experiments",
        f_min=f_min,
        f_max=f_max,
        seed=seed,
    )

    freq = _create_frequency_info(n_samples, fs, f_min, f_max)
    rng = np.random.default_rng(seed)

    n_freqs = freq.freqs.size

    # Create multisine in frequency domain
    dft_matrix = np.fft.fft(np.eye(nu)) / np.sqrt(nu)
    spectrum = np.zeros((n_freqs, nu, nu, n_experiments), dtype=complex)
    spectrum[freq.excited_bins] = dft_matrix[None, :, :, None]

    phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_experiments))
    spectrum *= np.exp(phase)[:, :, None, :]

    if independent_subexperiments:
        experiment_phase = 1j * rng.uniform(0, 2 * np.pi, (n_freqs, nu, n_experiments))
        spectrum *= np.exp(experiment_phase)[:, None, :, :]

    # Convert to time domain
    u = np.fft.irfft(spectrum, n=n_samples, axis=0)

    u = _ensure_requested_amplitude(u, amplitude)

    if n_experiments == 1:
        u = u[..., 0]

    return RandomPhaseMultisine(u=u, freq=freq, amplitude=amplitude, seed=seed)


def _validate_arguments(
    *,
    n_samples: int,
    fs: float,
    amplitude: float | tuple[float, ...],
    nu: int,
    additional_count: int,
    additional_count_name: str,
    f_min: float | None,
    f_max: float | None,
    seed: int,
) -> None:
    _validate_sampling_frequency(fs)
    _validate_strictly_positive_int(n_samples, "n_samples")
    _validate_strictly_positive_int(nu, "nu")
    _validate_strictly_positive_int(additional_count, additional_count_name)
    _validate_strictly_positive_int(seed, "seed")
    _validate_amplitude(amplitude, nu)

    if n_samples < MIN_NUMBER_OF_SAMPLES:
        msg = (
            f"The number of samples must be at least {MIN_NUMBER_OF_SAMPLES} to "
            f"allow for a meaningful frequency range, got n_samples={n_samples!r}."
        )
        raise ValueError(msg)

    _get_frequency_bins(n_samples, fs, f_min, f_max)


def _validate_sampling_frequency(fs: float) -> None:
    if isinstance(fs, bool) or not isinstance(fs, Real):
        msg = f"Sampling frequency must be a real number, got {type(fs).__name__} = {fs!r}."
        raise TypeError(msg)

    if not np.isfinite(fs) or fs <= 0:
        msg = f"Sampling frequency must be finite and strictly positive, got {fs!r}."
        raise ValueError(msg)


def _validate_amplitude(amplitude: float | tuple[float, ...], nu: int) -> None:
    if isinstance(amplitude, tuple):
        if len(amplitude) != nu:
            msg = f"amplitude must contain one value per input channel, got {amplitude!r}."
            raise ValueError(msg)
        if not all(isinstance(value, float) for value in amplitude):
            msg = f"amplitude must be a tuple of floats, got {amplitude!r}."
            raise TypeError(msg)
        if any(not np.isfinite(value) or value <= 0 for value in amplitude):
            msg = (
                f"amplitude must contain only finite, strictly positive values, "
                f"got {amplitude!r}."
            )
            raise ValueError(msg)
    elif not isinstance(amplitude, float):
        msg = f"amplitude must be a float or a tuple of floats, got {amplitude!r}."
        raise TypeError(msg)
    elif not np.isfinite(amplitude) or amplitude <= 0:
        msg = f"amplitude must be finite and strictly positive, got {amplitude!r}."
        raise ValueError(msg)


def _validate_strictly_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = (
            f"{name} must be a strictly positive integer, got {type(value).__name__} = "
            f"{value!r}."
        )
        raise TypeError(msg)
    if value <= 0:
        msg = f"{name} must be a strictly positive integer, got {value!r}."
        raise ValueError(msg)


def _create_frequency_info(
    n_samples: int,
    fs: float,
    f_min: float | None,
    f_max: float | None,
) -> FrequencyInfo:
    f_res = fs / n_samples
    f_min_bin, f_max_bin = _get_frequency_bins(n_samples, fs, f_min, f_max)

    n_freqs = n_samples // 2 + 1
    freqs = np.arange(n_freqs) * f_res
    excited_bins = np.arange(f_min_bin, f_max_bin + 1)
    non_excited_bins = np.setdiff1d(np.arange(n_freqs), excited_bins)

    return FrequencyInfo(
        fs=fs,
        f_res=f_res,
        f_min=f_min_bin * f_res,
        f_max=f_max_bin * f_res,
        freqs=freqs,
        excited_bins=excited_bins,
        non_excited_bins=non_excited_bins,
    )


def _get_frequency_bins(
    n_samples: int,
    fs: float,
    f_min: float | None,
    f_max: float | None,
) -> tuple[int, int]:
    """Calculate and validate the excited-bin bounds."""
    f_res = fs / n_samples
    f_min_bin = 1 if f_min is None else int(np.round(f_min / f_res))
    f_min_bin = max(f_min_bin, 1)  # ensure that DC is not excited
    f_max_bin = n_samples // 2 - 1 if f_max is None else int(np.round(f_max / f_res))

    if f_max_bin >= n_samples // 2:
        msg = (
            f"The maximum frequency must be less than the Nyquist frequency "
            f"(fs / 2), got f_max={f_max!r}, fs={fs!r}."
        )
        raise ValueError(msg)

    if f_min_bin > f_max_bin:
        msg = (
            f"The minimum frequency must be less than or equal to the maximum "
            f"frequency, got f_min={f_min!r}, f_max={f_max!r}."
        )
        raise ValueError(msg)

    return f_min_bin, f_max_bin


def _ensure_requested_amplitude(
    u: NDArray[np.floating[Any]],
    requested_amplitude: float | tuple[float, ...],
) -> NDArray[np.floating[Any]]:
    if isinstance(requested_amplitude, float):
        current_amplitude = _rms(u)
        return u * (requested_amplitude / current_amplitude)

    channel_axis = 1
    nu = u.shape[channel_axis]

    # Compute one RMS value per channel over samples and all trailing axes
    reduction_axes = (0, *range(channel_axis + 1, u.ndim))
    current_amplitudes = _rms(u, axis=reduction_axes, keepdims=True)

    # Keep the channel axis while broadcasting across every reduced axis
    n_trailing_axes = u.ndim - channel_axis - 1
    amplitude_shape = (1, nu) + (1,) * n_trailing_axes
    requested_amplitudes = np.asarray(requested_amplitude).reshape(amplitude_shape)
    return u * (requested_amplitudes / current_amplitudes)


def _rms(
    signal: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> NDArray[np.floating[Any]]:
    return np.sqrt(np.mean(np.abs(signal) ** 2, axis=axis, keepdims=keepdims))
