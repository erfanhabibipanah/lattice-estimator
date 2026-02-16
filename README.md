# LWE Parameter Sweep for PIR

This repository contains scripts to find LWE parameters for Private Information Retrieval (PIR) that achieve 128-bit security.

## Requirements

- SageMath with lattice-estimator
- Python 3

## Scripts

### 1. `lwe_pir_sweep.py` - Find All Passing Parameters

This script finds all (n, sigma) pairs that achieve 128-bit security for given parameters.

**How to run:**

```bash
# For q=2^32
python3 lwe_pir_sweep.py --log_N 36 --log_q 32

# For q=2^64
python3 lwe_pir_sweep.py --log_N 36 --log_q 64

# With custom delta
python3 lwe_pir_sweep.py --log_N 36 --log_q 64 --log_delta 20
```

**Options:**
- `--log_N` : Database size as power of 2 (36, 37, 38, or 39)
- `--log_q` : Ciphertext modulus as power of 2 (32 or 64)
- `--log_delta` : Correctness parameter (default: 40, meaning delta=2^-40)
- `--min_p` : Minimum p value to accept
- `--n_values` : Custom n values to test (e.g., "1024,1536,2048")
- `--output` : Output CSV file name

**Output:** CSV file with all passing (n, sigma) combinations marked with BEST column.

---

### 2. `lwe_pir_sweep_v2.py` - Find Minimum n for Target p

This script finds the minimum n value (with precision 1) for each target p value.

**How to run:**

```bash
# Find minimum n for different target p values
python3 lwe_pir_sweep_v2.py --log_N 36 --log_q 64 --log_delta 40

# With custom target p values
python3 lwe_pir_sweep_v2.py --log_N 36 --log_q 64 --target_p "16,18,20,22"

# With custom n search range
python3 lwe_pir_sweep_v2.py --log_N 36 --log_q 64 --n_min 1024 --n_max 3000
```

**Options:**
- `--log_N` : Database size as power of 2
- `--log_q` : Ciphertext modulus (32 or 64)
- `--log_delta` : Correctness parameter (default: 40)
- `--target_p` : Target p values as powers of 2 (e.g., "16,18,20,22" for 2^16, 2^18, etc.)
- `--n_min` : Minimum n to search (default: 1024)
- `--n_max` : Maximum n to search (default: 4096)
- `--output` : Output CSV file name

**Output:** CSV file with minimum n for each target p.

---

### 3. `lwe_simplepir_sweep.py` - SimplePIR Parameters

This script finds parameters specifically for SimplePIR paper verification.

**How to run:**

```bash
python3 lwe_simplepir_sweep.py
```

---

## Output CSV Files

The repository includes pre-computed results:

- `lwe_results_N36_q32.csv` - Results for N=2^36, q=2^32
- `lwe_results_N36_q64.csv` - Results for N=2^36, q=2^64
- `lwe_optimal_N{36-39}_q64_d{20,40}.csv` - Minimum n values for different configurations

## Parameters

| Parameter | Description |
|-----------|-------------|
| N | Database size (number of elements) |
| n | LWE dimension (affects security) |
| q | Ciphertext modulus |
| sigma | Noise standard deviation |
| p | Plaintext modulus (bits per element) |
| delta | Correctness parameter |

## Formula

The plaintext modulus p is calculated using:

```
p = sqrt(q / (sqrt(2) * sigma * N^(1/4) * sqrt(ln(2/delta))))
```

## Security

All results achieve 128-bit security against known LWE attacks using the lattice-estimator.
