# multisine

A minimal Python package for generating multisine excitation signals for frequency-domain system identification.

The multisine definitions implemented here follow *[System Identification: A Frequency Domain Approach, Second Edition][pintelon-schoukens]* by Rik Pintelon and Johan Schoukens.

## Status

This initial 0.1.x release is deliberately limited in scope.
It provides only random-phase multisine and orthogonal multisine generation.
Its API, validation, and feature set will grow in later releases.

## Install

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

[pintelon-schoukens]: https://doi.org/10.1002/9781118287422
