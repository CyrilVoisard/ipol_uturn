# U-Turn Segmentation with a Lower-Back Inertial Sensor

## General information

IPOL article: Voisard et al., *U-Turn Segmentation with a Lower-Back Inertial Sensor: an
Implementation of the Model of Salarian et al.*, Image Processing On Line (submitted).
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
U-turns detected in the recording (1 expected; a different value reveals a deviation from the
protocol) and the gyroscope drift that was removed.

The algorithm has four steps (Section 3 of the article):

1. integration of the angular velocity into a trajectory (heading), and removal of a linear
   drift estimated on the most linear window of the recording;
2. centering of the trajectory between its two levels and detection of its sign changes;
3. fitting window made of a fraction (60%) of each side of the turn, centering and orientation
   of the trajectory;
4. non-linear least-squares fit of the U-turn model of Salarian et al.;
   the U-turn is the transition interval of the fitted step.

The parameters of Table 1 of the article are gathered in `package/config.py`; the constants of
the drift estimation (window lengths, linearity score) are fixed in `package/uturn_detection.py`.

## Running the code

Python 3.9 or later.

```
pip install -r requirements.txt
python main.py -i0 <file>_lb.txt -freq 100 -unit rad/s
```

Arguments:

- `-i0`: text file of the lower-back sensor (header lines are skipped; the table must contain
  the columns `PacketCounter` and `Gyr_X`, other columns are ignored);
- `-freq`: acquisition frequency in Hz;
- `-unit`: unit of `Gyr_X`, `rad/s` (default) or `deg/s`.

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

## Application to a public data set (Section 4 of the article)

```
python validation/run_real_data_study.py DATA_FOLDER [OUTPUT_FOLDER] [--group HS]
```

segments every lower-back recording of the data set of Voisard et al. (Scientific Data, 2025:
`*_raw_data_LB.txt` and `*_meta.json`; the IPOL 2023 layout `*_lb.txt` is also accepted) and
writes in OUTPUT_FOLDER:

- `real_runs.csv` and `real_summary.txt`: boundaries, outputs and indicators of every
  recording, summarised overall and per group;
- `visual_check_[group].pdf`: trajectory, fitted model and boundaries of every recording, for the
  visual inspection (recordings with a warning are titled in red; the boundaries given in the
  metadata are dotted);
- `fig_example_result.pdf` and `fig_example_construction.pdf`: Figures 4 and 3 of the article.

## License

GNU General Public License v3.0, see `LICENSE`.
