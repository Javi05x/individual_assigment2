# Optimised Approaches and Sparse Matrix Multiplication

This repository contains the implementation, validation, benchmark data,
figures and final report for Individual Assignment 2.

## Scope

The project evaluates:

- Basic dense matrix multiplication using the `i-j-k` loop order.
- Row-oriented dense multiplication using the `i-k-j` loop order.
- Tiled dense multiplication with a block size of 32.
- A custom pure-Python Compressed Sparse Row implementation.
- SciPy CSR matrix multiplication.
- Dense-versus-sparse behaviour at controlled sparsity levels.
- The real-world `mc2depi` sparse matrix.

## Experimental Design

The final synthetic experiments use:

- Matrix sizes: 64, 128, 256 and 512.
- Input sparsity levels: 50%, 70%, 90%, 95% and 99%.
- One warm-up execution per configuration.
- Five measured executions per configuration.
- Deterministic and matched logical inputs.
- Median and interquartile range as summary statistics.
- Isolated processes for RSS memory measurements.

## Repository Structure

- `data/mc2depi/`: real-world Matrix Market dataset.
- `data/generated/`: auxiliary generated data.
- `python/src/matrix/benchmark/`: benchmark and processing scripts.
- `python/src/matrix/dense/`: dense implementations and generators.
- `python/src/matrix/sparse/`: CSR implementations and generators.
- `python/src/matrix/plots/`: plot-generation scripts.
- `python/test/matrix/`: correctness tests.
- `results/raw/`: individual benchmark measurements.
- `results/processed/`: processed statistical summaries.
- `plots/`: final report figures.
- `report/individual_assignment_2.pdf`: final report.
- `ENVIRONMENT.md`: experimental environment and limitations.
- `requirements.txt`: Python dependencies.

## Installation

Create and activate a virtual environment on Windows with Git Bash:

    python -m venv .venv
    source .venv/Scripts/activate

Install the dependencies:

    python -m pip install -r requirements.txt

## Correctness Validation

Run the complete test suite with:

    python -m pytest python/test/matrix -v

The validated project contains 102 passing tests.

## Final Benchmarks

Dense execution-time benchmark:

    python -m python.src.matrix.benchmark.benchmark_dense --profile full --output results/raw/dense_time_full.csv

Dense memory benchmark:

    python -m python.src.matrix.benchmark.benchmark_dense_memory --profile full --output results/raw/dense_memory_full.csv

Dense-versus-sparse execution-time benchmark:

    python -m python.src.matrix.benchmark.benchmark_sparse --profile full --output results/raw/sparse_time_full.csv

Dense-versus-sparse memory benchmark:

    python -m python.src.matrix.benchmark.benchmark_sparse_memory --profile full --output results/raw/sparse_memory_full.csv

Process the final measurements:

    python -m python.src.matrix.benchmark.process_results

Generate the final figures:

    python -m python.src.matrix.plots.plot_results

## Real-World Dataset

The project uses the `mc2depi` matrix as a real-world sparse case study.

The dataset has:

- 525,825 rows.
- 525,825 columns.
- 2,100,225 stored nonzero entries.
- Approximately 99.99924040% sparsity.

The dataset is processed directly in sparse form. A complete dense
representation is never created.

The final case study includes:

- Sparse matrix-vector multiplication.
- Sparse matrix-matrix multiplication using `mc2depi x mc2depi`.

## Final Report

The final PDF is available at:

    report/individual_assignment_2.pdf

## Experimental Environment

The experiments were performed using CPython 3.13.12 on Microsoft Windows
11 Pro. Complete hardware, software, benchmark and measurement details are
provided in `ENVIRONMENT.md`.

## AI-Assistance Declaration

Generative AI assistance was used for planning, code review, test design,
debugging, benchmark-protocol design and report structuring. All
implementations were executed and validated locally. All reported numerical
results were obtained from the preserved benchmark measurements.
