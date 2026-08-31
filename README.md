# Depth resolvability in the Fars Arc, Zagros

Analysis code for:

> Khalili, M., and A. Fotoohi. What do ISC Bulletin depths resolve?
> Reporting-agency effects and network geometry in the Fars Arc, Zagros.

The code reads International Seismological Centre (ISC) Bulletin arrival and
origin exports for a rectangle covering the Fars Arc, and quantifies four
things: how far the reported depth column is quantized, how much of the pooled
depth signal is carried by the changing mix of reporting agencies, what the
azimuthal gap becomes when it is recomputed from stations inside a series of
epicentral-distance cutoffs, and what fraction of events has a recording
geometry capable of constraining focal depth.

Every number quoted in the manuscript is produced by these scripts and written
to `output/results.json`. Nothing is recomputed at figure time, and
`scripts/04_check_manuscript.py` compares the manuscript text against that file
scripts/06_depth_distance_regression.py  fit depth against nearest-station distance
`scripts/05_depth_sensitivity.py` computes the formal depth sensitivity of the recorded and proposed station geometries, writing `output/depth_sensitivity.json` and `output/table_s2_depth_sensitivity.csv` (Table S2 of the supplement).
claim by claim.

## Requirements

Python 3.9 or later, plus NumPy, pandas, Matplotlib and PyYAML. `requests` is
needed only to download the ISC exports; the analysis runs from the released
data files without it. There is no SciPy dependency: the rank correlation and
its p-value are implemented in `src/stats.py` and are covered by tests.

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate zagros-depth-resolvability
```

## Data

The analysis expects one ISC ARRIVAL:ASSOCIATED CSV export per year in a data
directory, named `arr_<year>.txt`. To retrieve them:

```bash
python scripts/00_download_isc.py --output data --start 2010 --end 2026
```

The search rectangle and the year range come from `config.yaml`. Requests are
spaced out and already downloaded years are skipped, so the script can be
interrupted and restarted.

The ISC Bulletin is cited separately from this software; see `CITATION.cff`.

## Running the analysis

```bash
python scripts/01_run_analysis.py  --input data   --output output
python scripts/02_make_figures.py  --results output --output figures
python scripts/03_depth_flags.py   --input data   --output output
python scripts/04_check_manuscript.py --results output
```

`01_run_analysis.py` writes

| File | Contents |
| --- | --- |
| `output/results.json` | every reported statistic, with the configuration and the package versions used |
| `output/events_derived.csv` | one row per event, with the recording geometry attached |
| `output/stations_derived.csv` | one row per station |
| `output/input_manifest.csv` | size, SHA-256 and row diagnostics for each input file |

`03_depth_flags.py` parses the ISF2 origin blocks, which carry the depth fix
flags and the reported depth uncertainties that the CSV export omits, and
writes `output/depth_flag_diagnostics.json` and
`output/parsed_isf_origins_full.csv`.

`04_check_manuscript.py` re-evaluates every numerical claim in the manuscript
against `output/results.json` and exits non-zero if any of them disagrees.

## Figures

`02_make_figures.py` writes six figures, each in PDF, PNG, EPS and TIFF, all
from the same figure object so the versions cannot disagree:

| Name | Subject |
| --- | --- |
| `fig02_station_event_map` | epicenters and recording stations in the study area |
| `fig03_depth_quantization` | the distribution of reported depth values |
| `fig04_agency_confounding` | pooled against per-agency depth medians by year |
| `fig05_gap_versus_cutoff` | azimuthal gap as a function of distance cutoff |
| `fig06_network_capability` | nearest-station distance and station count against gap |
| `fig07_recent_era` | agency composition and depth style after the reviewed period |

Figure 1 of the manuscript is a tectonic setting map that is not generated
here.

Output resolution and the colour palette are read from the `figures` block of
`config.yaml`. The palette was checked for separation under normal colour
vision and under the common colour-vision deficiencies; the script does not
override it.

## Tests

```bash
python tests/run_tests.py     # 43 tests, no third-party test runner needed
python -m pytest tests        # equivalent, if pytest is installed
```

The tests cover the geometry functions against known answers, the rank
correlation against an independent numerical integration, the ISC reader
against a synthetic export that reproduces both field counts seen in the real
files, and the analysis layer against constructed cases.

Two groups of tests exist because the corresponding errors would be silent
rather than fatal. The first checks that the azimuthal gap at a distance cutoff
is computed only from stations inside that cutoff. The second checks that the
reviewed and unreviewed periods are separated before any statistic is taken,
since pooling them would change the answers without raising anything.

## Layout

```
config.yaml                  all parameters used anywhere in the analysis
src/geometry.py              spherical distance, azimuth, azimuthal gap, take-off angle
src/stats.py                 rank correlation and the incomplete beta function
src/isc_io.py                ISC export reader and provenance manifest
src/analysis.py              every reported statistic
scripts/00_download_isc.py   retrieve the ISC exports
scripts/01_run_analysis.py   run the analysis, write results.json
scripts/02_make_figures.py   draw the figures from results.json
scripts/03_depth_flags.py    parse the ISF2 origin blocks
scripts/04_check_manuscript.py  verify the manuscript against results.json
tests/                       test suite
```

`config.yaml` is the only place numerical constants are declared. The source
files contain none of their own, so a parameter cannot be changed in one place
and left stale in another.

## License

MIT, see `LICENSE`. If you use this code, please cite both the software and the
paper; `CITATION.cff` has the details.
