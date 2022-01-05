# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.2.3] - 2022-01-05

### Added

- This CHANGELOG
- `requirements-dev.txt`
- Documentation
- Upload built conda environment as an artifact
- Notebook rendering to Github actions, including hash-based artifact checks
- Initial benchmarking tooling, still in progress

### Changed

- Moved publishing steps into separate workflows
- Deprecated imports from `odc.index.*`
- Removed `.units` attribute from `.time` axis for better inter-op with `.to_zarr`, `.to_netcdf`

### Fixed

- Handling of STAC Items with only partial `proj` data
- Typos in documentation

## [v0.2.2] - 2021-10-25

### Added

- Binder launcher to README
- Another USGS STAC example for Landsat SR
- Documentation

### Changed

- Cleaned up test fixtures
- Relaxed `is_raster_data` check
- Force data band decision for explicitly configured bands
- Moved constansts in to global scope

## [v0.2.1] - 2021-10-18

Initial release as a standalone project.
Previously, this project was part of https://github.com/opendatacube/odc-tools.
