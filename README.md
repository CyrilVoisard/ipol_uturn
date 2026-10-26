# U-Turn Segmentation in a Go-and-Back Walking Test with a Lower-Back IMU

## General information

IPOL article: Voisard et al., *U-Turn Segmentation with a Lower-Back Inertial Sensor: an
Implementation of the Turning Model of Salarian et al.*, Image Processing On Line (submitted).
IPOL demo: (link given once the article is published)
Contact: cyril.voisard@gmail.com

Please cite the IPOL article and the article that introduced the turning model:

- A. Salarian, C. Zampieri, F. B. Horak, P. Carlson-Kuhta, J. G. Nutt and K. Aminian,
  *Analyzing 180° turns using an inertial system reveals early signs of progress in Parkinson's
  disease*, IEEE EMBC 2009, pp. 224–227, https://doi.org/10.1109/IEMBS.2009.5333970

## What the code does

The input is the angular velocity about the X axis (vertical axis when standing) of an inertial
sensor placed on the lower back (L5), recorded during a test made of a standing period, a
straight walk, a U-turn, a straight walk back and a standing period. The code returns the first
and last samples of the U-turn, its duration, amplitude, direction and peak angular velocity,
and three indicators: the coefficient of determination of the model fit (R²), the number of
half-turns detected in the recording (1 expected; a different value reveals a deviation from the
protocol) and the gyroscope drift that was removed.

The algorithm has four steps (Section 3 of the article):

1. integration of the angular velocity into a heading, and removal of a linear drift estimated
   on the most linear window of the recording;
2. centring of the heading between its two levels and detection of its sign changes;
3. fitting window made of the half of each side of the turn adjacent to it, centring and
   orientation of the heading;
4. non-linear least-squares fit of the turning model of Salarian et al.;
   the U-turn is the transition interval of the fitted step.

All numerical parameters are gathered, with their meaning, in `package/config.py`.

## Running the code

Python 3.9 or later.

```
pip install -r requirements.txt
python main.py -i0 example_datasets/<file>_lb.txt -freq 100 -unit rad/s
```

Arguments:

- `-i0`: text file of the lower-back sensor, as exported by XSens MT Manager (header lines are
  skipped; the table must contain the columns `PacketCounter` and `Gyr_X`, other columns are
  ignored);
- `-freq`: acquisition frequency in Hz;
- `-unit`: unit of `Gyr_X`, `rad/s` (XSens default) or `deg/s`;
- `-legacy`: `1` to use the rules of the original implementation (see below), `0` by default.

Outputs, written in the current folder: `uturn.png` (result), `uturn_construction.png`
(intermediate steps), `uturn.txt` (results and indicators) and `uturn.json` (results,
indicators and parameters). If the U-turn cannot be segmented, the reason is written in
`demo_failure.txt`.

From Python:

```python
from package import import_data, uturn_detection
gyr = import_data.import_lower_back("trial_lb.txt", freq=100, gyr_unit="rad/s")
res = uturn_detection.detect_uturn(gyr, freq=100)
print(res.start, res.end, res.duration, res.amplitude, res.r2, res.n_half_turns)
```

## Legacy rules

Clinical studies published before the IPOL article used a first implementation of the same
algorithm (kept verbatim in `validation/legacy/uturn_detection_sal.py`). The present
implementation replaces three rules of Steps 2 and 3 (listed in `package/config.py` and in
Section 5 of the article). `-legacy 1` (or `config.LEGACY = True`) reproduces the original
rules; `validation/check_legacy_equivalence.py` verifies on random recordings that the boundaries
are then identical to those of the original code. The present implementation is also about
1000 times faster (cumulative sums instead of one regression per window).

## Validation

- `validation/run_synthetic_study.py`: accuracy on synthetic recordings with an exact ground
  truth (robustness to noise, drift, trunk oscillations, turn duration and position, sampling
  frequency, amplitude; turn profiles; protocol deviations; fitting window).
- `validation/run_synthetic_sensitivity.py`: sensitivity of the boundaries to every parameter,
  on synthetic recordings.
- `validation/check_legacy_equivalence.py`: equivalence of the legacy rules with the original code.
- `validation/make_figures.py`: figures and tables of the article.
- `validation/run_real_data_study.py DATA_FOLDER`: on the public data set of Voisard et al.
  (IPOL 2023, https://doi.org/10.5201/ipol.2023.497), consistency with the published U-turn
  boundaries, distribution of the outputs and indicators, and sensitivity to every parameter.
- `validation/synthetic.py`: generator of synthetic recordings.

## License

GNU General Public License v3.0, see `LICENSE`.
