# Reproducing this study

## Source data
    file    REGINA_ABABIO.xlsx
    bytes   789758
    sha256  99ffc70ff7a8bb490c429384574d2022086c70e85b76acd0c9d9fa7fe9740fde

The raw GMet records are not redistributed here. Obtain them from the Ghana
Meteorological Agency and verify the digest above before running. If the digest
differs you are working from a different export and results may not match.

## Environment
`requirements.txt` pins every library version; `environment.json` additionally
records the interpreter, platform, core count and memory of the machine used.
All notebooks are CPU-only.

## Determinism
A single seed (42) is set in notebook 01 and read from `config.json` by every
later notebook. numpy, random and torch are all seeded from it, and torch runs
with deterministic algorithms enabled.

## Run order
Notebooks are numbered and must run in sequence. Each asserts that the artefacts
of its predecessors exist before doing any work, so running out of order fails
loudly rather than producing wrong numbers.

## Parameters
Every analytic choice lives in `config.json`: study period, partition
boundaries, look-back length, minimum days per month, interpolation gap limit,
sentinel codes and physical bounds. Nothing is hard-coded in a notebook.

## Numbers in the thesis
Every figure and table is written to `outputs/`. No number is transcribed by
hand. `outputs/all_paper_numbers.json`, produced by the final notebook, is the
single source for the written chapters.
