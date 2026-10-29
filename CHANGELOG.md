# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-10-29

### Added

- Command-line sentiment prediction for one or more Turkish texts.
- Confidence scores, UTF-8 file input, and optional JSON output.
- Shared text preprocessing for consistent training and inference.
- Correct handling of uppercase Turkish `I` and `İ` during normalization.
- Unit tests for preprocessing and prediction behavior.
- Reproducible model loading with the persisted scikit-learn version.
- MIT License for the project source code.

## [2.0.0] - 2026-06-04

### Added

- Iterative training and evaluation across five TF-IDF feature counts.
- Per-model reports and a model comparison summary.

## [1.0.0] - 2026-06-04

### Added

- Initial Turkish sentiment analysis workflow.

[2.1.0]: https://github.com/barisakk00/turkish-sentiment-analysis/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/barisakk00/turkish-sentiment-analysis/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/barisakk00/turkish-sentiment-analysis/releases/tag/v1.0.0
