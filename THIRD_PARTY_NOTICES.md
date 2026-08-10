# Third-Party Notices

## Scope of the Current Release

The current release snapshot contains the core KDS-Former implementation
released under the MIT License (see `LICENSE`) and does not include
third-party baseline source code.

## Runtime Dependencies

The following packages are installed separately via `pip`/`conda` (see
`requirements.txt` and `environment.yml`) and are not included in this
repository. Each remains under its own upstream license:

| Package | License |
|---|---|
| PyTorch  | BSD-3-Clause |
| NumPy    | BSD-3-Clause |
| tqdm     | MIT / MPL-2.0 |
| pytest   | MIT |

Refer to each project's official repository for the authoritative license
text.

## Baseline Methods Referenced in the Paper

The manuscript compares KDS-Former against the following published methods:

| Baseline | Reference | Official repository |
|---|---|---|
| ST-GCN | Yan, Xiong & Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018 | https://github.com/yysijie/st-gcn |
| Min et al. (VIPL-SLP) | Min et al., "A Closer Look at Skeleton-Based Continuous Sign Language Recognition", ICCVW 2025, DOI 10.1109/ICCVW69036.2025.00515 | https://github.com/VIPL-SLP/MSLR_ICCV2025 |
| TEMPO | Hassan & Alsayad, "TEMPO at SignEval 2026" (1st place, MSLR 2026 Track 1), CVPRW 2026 | https://github.com/AhmedMo1242/TEMPO |

Their implementations are not included in this repository. To reproduce
those comparisons, obtain the respective official implementations from
their original authors' repositories under those repositories' own license
terms.
