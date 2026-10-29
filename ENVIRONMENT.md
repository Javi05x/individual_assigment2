# Experimental Environment

## Hardware

The experiments were executed on the following system:

- Manufacturer: HP
- Model: Victus by HP Gaming Laptop 15-fa2xxx
- Processor: 13th Gen Intel(R) Core(TM) i5-13420H
- Physical CPU cores: 8
- Logical processors: 12
- Maximum clock speed reported by Windows: 2100 MHz
- Installed physical memory: 15.64 GiB
- System architecture: 64-bit

The processor frequency is the value reported by the Windows `Win32_Processor.MaxClockSpeed` property. It is included as system metadata and should not be interpreted as a constant frequency maintained throughout every benchmark run.

## Operating System

- Operating system: Microsoft Windows 11 Pro
- Version: 10.0.26300
- Build number: 26300
- Active power plan: Balanced

The experiments were run from Git Bash inside a Python virtual environment.

## Python Environment

- Python: 3.13.12
- Python implementation: CPython
- NumPy: 2.5.4
- SciPy: 1.18.1
- pandas: 3.1.0
- Matplotlib: 3.11.2
- psutil: 7.2.2
- pytest: 9.2.0

The project dependencies were installed in the local `.venv` virtual environment.

## Benchmark Conditions

The synthetic time benchmarks used:

- One warm-up execution per algorithm and configuration.
- Five measured executions per configuration.
- `time.perf_counter_ns()` as the high-resolution timer.
- Deterministic input generation with recorded seeds.
- Identical logical input matrices for all algorithms in each comparison.
- Input generation and representation conversion outside the timed multiplication region.
- Median and interquartile range as the principal summary statistics.
- Preservation of every individual measurement in CSV format.
- Correctness validation before the measured executions.
- Checksums and output nonzero counts during the measured experiments.

The order of the algorithms was changed reproducibly between measured runs in the time benchmarks to reduce the effect of consistently executing one implementation first or last.

## Memory Measurement Protocol

Process memory was measured using the Resident Set Size reported by `psutil`.

For the synthetic memory experiments:

- Every measurement was executed in a fresh Python process.
- Five independent processes were used per algorithm and configuration.
- Process RSS was sampled every 0.001 seconds during multiplication.
- The initial RSS, representation RSS, pre-multiplication RSS, observed peak RSS and final RSS were recorded.
- Both the multiplication peak delta and the total peak delta were retained.

RSS represents the memory resident in the complete Python process. It includes the interpreter, imported modules, allocator behaviour and matrix data. Therefore, small RSS differences between algorithms should not be interpreted as exact differences in matrix storage.

## Dense Benchmark Configuration

The final dense benchmark used square matrices with the following side lengths:

- 64
- 128
- 256
- 512

The matrix values were integers drawn from a discrete uniform distribution over the interval `[-10, 10]`.

The following implementations were evaluated:

- Basic dense multiplication using the `i-j-k` loop order.
- Row-oriented dense multiplication using the `i-k-j` loop order.
- Tiled dense multiplication using the `i-k-j` loop order and a tile size of 32.

## Synthetic Sparse Benchmark Configuration

The final synthetic sparse benchmark used:

- Matrix sizes: 64, 128, 256 and 512.
- Requested sparsity levels: 50%, 70%, 90%, 95% and 99%.
- Nonzero integer values drawn from `[-10, 10]`, excluding zero.
- Exact deterministic selection of nonzero positions without replacement.

The following representations and implementations were compared:

- Row-oriented dense multiplication.
- A custom pure-Python Compressed Sparse Row implementation.
- SciPy Compressed Sparse Row multiplication.

The input matrices were generated once in logical dense form for each configuration and then converted to the required representations. Consequently, every algorithm received the same logical values and nonzero positions.

## Real-World Dataset

The real-world case study used the `mc2depi` matrix in Matrix Market coordinate format.

Observed dataset properties:

- Rows: 525,825
- Columns: 525,825
- Stored nonzero entries: 2,100,225
- Logical entries: 276,491,930,625
- Density: approximately 0.00075960%
- Sparsity: approximately 99.99924040%
- Stored-value range: -3,559 to 2,025
- Stored explicit zeros after loading: 0
- SciPy CSR array storage: approximately 26.041 MiB
- Theoretical dense `int64` value storage: approximately 2,060.025 GiB

The theoretical dense storage figure includes only one 64-bit integer per logical matrix position. It does not include container, row, index or Python-object overhead.

The dataset was loaded directly into a SciPy CSR representation. A complete dense representation of `mc2depi` was never created.

Two real-world operations were evaluated:

- Sparse matrix-vector multiplication.
- Sparse matrix-matrix multiplication using `mc2depi × mc2depi`.

For the final sparse matrix-matrix experiment, result consistency across all five measured runs was verified using:

- Output nonzero count.
- Output storage size.
- Sum of output values.
- Sum of the absolute output values.
- SHA-256 over the complete canonical CSR data, column-index and row-pointer arrays.

## Experimental Limitations

The active Windows power plan was Balanced rather than a fixed high-performance plan. Processor frequency, background operating-system activity, thermal conditions and system caching may therefore have introduced run-to-run variation.

The benchmark results describe this specific hardware and software environment. They should not be interpreted as universal performance rankings for all Python, NumPy, SciPy or processor configurations.

The memory monitor used periodic RSS sampling and may not detect allocations whose lifetime is shorter than the sampling interval. The reported peak values are therefore the highest observed RSS values rather than a guarantee of the absolute instantaneous memory peak.
