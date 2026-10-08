# multisine

A minimal Python package for generating multisine excitation signals for frequency-domain system identification.

The multisine definitions implemented here follow *[System Identification: A Frequency Domain Approach, Second Edition][pintelon-schoukens]* by Rik Pintelon and Johan Schoukens.

## Status

This initial 0.1.x release is deliberately limited in scope.
It provides only random-phase multisine and orthogonal multisine generation.
Its API, validation, and feature set will grow in later releases.

## Installation

Requires Python 3.10 or later.
The only runtime dependency is NumPy (1.24 or later).

Install with `pip`:

```bash
pip install multisine
```

Or add it to a `uv` project:

```bash
uv add multisine
```

## Example

```python
from multisine import random_phase_multisine, random_phase_orthogonal_multisine

multisine = random_phase_multisine(
    n_samples=1024,
    fs=100.0,
    nu=2,
    amplitude=1.0,
)
orthogonal_multisine = random_phase_orthogonal_multisine(
    n_samples=1024,
    fs=100.0,
    nu=2,
    amplitude=1.0,
)

print(multisine.u.shape)  # (1024, 2)
print(orthogonal_multisine.u.shape)  # (1024, 2, 2)
```

## Related packages

Looking for:

- nonparametric estimation of the best linear approximation? See [best-linear-approximation](https://github.com/merijnfloren/best-linear-approximation).
- parametric state-space models, linear or nonlinear? See [freq-statespace](https://github.com/merijnfloren/freq-statespace).

[pintelon-schoukens]: https://doi.org/10.1002/9781118287422

<!-- BEGIN: reusable-result capability description -->
## For AI-assisted method selection

<!--
Maintained for the CoMoDO reusable result.
Template: comodo-reusable-result/templates/capability-section.md
This section is read by the `find-methods` skill to match this repository
against an external user's application. Keep it accurate; update it when
examples are added or removed.
-->

### Identity

- **One-line summary:** Generate repeatable, band-limited random-phase multisine signals for exciting a plant during frequency-domain system identification, including orthogonal signals for multi-input experiments.
- **Role:** method library
- **Maturity:** validated on lab hardware
- **Maintainer / lab:** Merijn Floren / KU Leuven MECO
- **Licence / availability:** MIT; Python package on PyPI as `multisine`

### Physical setup

<!-- Use-case demonstrators only. Method libraries and middleware: write `n/a`. -->

- **System under control:** n/a — signal-generation library; it does not control a plant.
- **Actuation:** n/a — returns sampled NumPy arrays; the user supplies their own I/O and units.
- **Sensing:** n/a — the library does not acquire measurements.
- **Sample rates:** n/a — the caller selects the sampling frequency `fs` in Hz and number of samples; signal frequency resolution is `fs / n_samples`.
- **Hardware & fieldbus:** n/a — no hardware or fieldbus interface.
- **Scale of the dynamics:** n/a — the caller selects the excitation band with `f_min` and `f_max`, below Nyquist.

### Problems this repository can help with

- Create a repeatable, band-limited excitation signal to identify a system's frequency response from measured input-output data (random-phase multisine).
- Excite several plant inputs while separating their frequency-response contributions and obtaining a well-conditioned frequency-response matrix estimate in the presence of noise and nonlinear distortions (random-phase orthogonal multisine).

### Examples

<!--
One row per runnable example. Entry point = the file someone opens first.
Keep this table complete: an example missing here is invisible to the tool.
-->

| Example | What it demonstrates | Entry point | Sim | Hardware |
| --- | --- | --- | --- | --- |
| Basic random-phase multisine | Generate a multi-input, RMS-scaled random-phase multisine as a NumPy array. | `README.md` — Example | Yes | n/a — no I/O support |
| Orthogonal multisine | Generate a multi-input orthogonal multisine for frequency-response matrix estimation. | `README.md` — Example | Yes | n/a — no I/O support |

### Preconditions

<!--
What must be true of the user's application before any of this transfers.
Be concrete: data you must be able to collect, signals you must be able to
inject, things you must be able to measure, compute you must have.
-->

- For a real-plant experiment, the user must be able to apply a sampled, periodic excitation signal to one or more plant inputs and measure the resulting response. The same signals may instead be used in a simulation identification experiment.
- The user must choose a sampling frequency, record length, and excitation band appropriate to the dynamics and their I/O system, and provide the interface that converts the returned NumPy array to actuator commands.
- The requested time-domain RMS amplitude is met by the generator. The user must independently check peak amplitude against actuator, plant, and safety constraints.

### Not suitable when

<!--
The honesty valve. Concrete situations where a user should look elsewhere,
and, where you know one, the name of what they should look at instead.
-->

- A real plant cannot safely receive the excitation because of actuator limits, plant limits, or safety constraints. Assess those limits before using the generated signal; this package does not enforce them.
- You need the package itself to communicate with hardware, schedule real-time output, or acquire measurements; it only generates NumPy arrays.
- You need crest-factor optimization or automatic peak-constraint-aware excitation design; neither is implemented.

### Dependencies beyond this repository

| Dependency | Why it is needed | Hard requirement? |
| --- | --- | --- |
| Python 3.10 or later | Runtime environment. | Yes |
| NumPy 1.24 or later | Signal generation and returned arrays. | Yes |
| Plant I/O and measurement system | Required only to apply signals and collect data in a real-plant experiment; supplied by the user. | Yes for real-plant use; no for simulation-only use |

### Where to look deeper

<!-- Files and directories worth reading, with one line each on what is in them. -->

- `src/multisine/_multisine.py` — public generators, output shapes, frequency-band selection, and argument validation.
- `tests/multisine/test_multisine.py` — checks RMS scaling, reproducibility, and orthogonality at excited frequencies.
- `README.md` — installation and minimal usage examples.

<!-- END: reusable-result capability description -->
