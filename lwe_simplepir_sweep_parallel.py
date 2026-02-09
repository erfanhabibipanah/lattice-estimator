#!/usr/bin/env python3
"""
Optimized parallel LWE parameter sweep for SimplePIR.
Uses multiprocessing to run security estimates in parallel.
"""

import sys
import math
import csv
import os
from multiprocessing import Pool, cpu_count
from functools import lru_cache

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CSV_FILE = "lwe_simplepir_all_results.csv"

LOG_N = 38
N = 2 ** LOG_N
m = int(math.ceil(math.sqrt(N)))
DELTA = 2 ** -40
TARGET_SECURITY = 128

LOG_Q_32 = 32
LOG_Q_64 = 64

# Use more cores for parallel processing
NUM_WORKERS = max(1, cpu_count() - 1)
# Increase jobs for lattice-estimator internal parallelism
ESTIMATOR_JOBS = 4

all_results = []

def compute_p(q, sigma):
    ln_term = math.log(2 / DELTA)
    denom = math.sqrt(2) * sigma * (N ** 0.25) * math.sqrt(ln_term)
    p_sq = q / denom
    return int(math.floor(math.sqrt(p_sq))) if p_sq > 0 else 0

def compute_sigma_for_p(q, target_p):
    ln_term = math.log(2 / DELTA)
    sigma = q / (math.sqrt(2) * (target_p ** 2) * (N ** 0.25) * math.sqrt(ln_term))
    return sigma

def get_security_single(args):
    """Worker function for parallel execution."""
    n, log_q, sigma, source = args
    try:
        from estimator import LWE, ND
        from sage.all import oo

        q = 2 ** log_q
        params = LWE.Parameters(
            n=n, q=q,
            Xs=ND.DiscreteGaussian(sigma),
            Xe=ND.DiscreteGaussian(sigma),
            m=m
        )
        result = LWE.estimate(params, jobs=ESTIMATOR_JOBS)

        min_bits = float("inf")
        for _, data in result.items():
            rop = data.get("rop", oo)
            if rop != oo:
                bits = math.log2(float(rop))
                if bits < min_bits:
                    min_bits = bits

        p = compute_p(q, sigma)
        log2_p = math.log2(p) if p > 0 else 0
        passes = min_bits >= TARGET_SECURITY

        return {
            'n': n,
            'log_q': log_q,
            'q': q,
            'sigma': sigma,
            'p': p,
            'log2_p': log2_p,
            'security': min_bits,
            'passes': 'YES' if passes else 'NO',
            'source': source
        }
    except Exception as e:
        return {
            'n': n, 'log_q': log_q, 'q': 2**log_q, 'sigma': sigma,
            'p': 0, 'log2_p': 0, 'security': 0, 'passes': 'ERROR',
            'source': f"{source}_error: {str(e)}"
        }

def parallel_sweep(tasks):
    """Run multiple security estimates in parallel."""
    print(f"  Running {len(tasks)} estimates with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(get_security_single, tasks)
    return results

def sweep_sigma_for_n_parallel(n_values, log_q, sigma_values, source_prefix):
    """Sweep all (n, sigma) combinations in parallel."""
    tasks = []
    for n in n_values:
        for sigma in sigma_values:
            tasks.append((n, log_q, sigma, source_prefix))

    results = parallel_sweep(tasks)

    for r in results:
        all_results.append(r)
        status = "PASS" if r['passes'] == 'YES' else "FAIL"
        print(f"    n={r['n']}, sigma={r['sigma']:.2f}, p={r['p']:,} (2^{r['log2_p']:.1f}), security={r['security']:.1f} [{status}]")

    return results

def find_min_n_binary_parallel(log_q, target_p, n_low, n_high):
    """Binary search with parallel fine search."""
    q = 2 ** log_q
    sigma_needed = compute_sigma_for_p(q, target_p)

    print(f"\n  Binary search for min n with p>={target_p:,} (sigma={sigma_needed:.2f})")
    print(f"  Search range: [{n_low}, {n_high}]")

    best_n = None

    # Coarse binary search (sequential for correct logic)
    while n_high - n_low > 64:
        n_mid = (n_low + n_high) // 2
        n_mid = (n_mid // 8) * 8

        result = get_security_single((n_mid, log_q, sigma_needed, f"binary_p{target_p}"))
        all_results.append(result)
        sec = result['security']

        status = "PASS" if sec >= TARGET_SECURITY else "FAIL"
        print(f"    n={n_mid}, security={sec:.1f} [{status}]")

        if sec >= TARGET_SECURITY:
            best_n = n_mid
            n_high = n_mid
        else:
            n_low = n_mid

    # Fine search in parallel
    fine_tasks = []
    for n in range(n_low, n_high + 1, 8):
        fine_tasks.append((n, log_q, sigma_needed, f"fine_p{target_p}"))

    if fine_tasks:
        print(f"  Fine search [{n_low}, {n_high}] with {len(fine_tasks)} values in parallel...")
        fine_results = parallel_sweep(fine_tasks)
        for r in fine_results:
            all_results.append(r)
            if r['security'] >= TARGET_SECURITY and (best_n is None or r['n'] < best_n):
                best_n = r['n']

    return best_n, sigma_needed

def write_csv():
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n', 'log_q', 'q', 'sigma', 'log_N', 'N', 'm',
            'delta', 'log_delta', 'p', 'log2_p', 'security_bits',
            'passes_128bit', 'source'
        ])
        for r in all_results:
            writer.writerow([
                r['n'], r['log_q'], r['q'], f"{r['sigma']:.2f}",
                LOG_N, N, m, f"{DELTA:.2e}", -40,
                r['p'], f"{r['log2_p']:.1f}", f"{r['security']:.1f}",
                r['passes'], r['source']
            ])
    print(f"\nAll {len(all_results)} results saved to {CSV_FILE}")

def main():
    print("=" * 70)
    print("SimplePIR LWE Parameter Sweep - PARALLEL VERSION")
    print("=" * 70)
    print(f"Database size:    N = 2^{LOG_N} = {N:,}")
    print(f"LWE samples:      m = sqrt(N) = {m:,}")
    print(f"Correctness:      delta = 2^-40 = {DELTA:.2e}")
    print(f"Security target:  {TARGET_SECURITY} bits")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Estimator jobs:   {ESTIMATOR_JOBS}")
    print("=" * 70)

    # q = 2^32 - parallel sweep for key n values
    print(f"\n{'='*70}")
    print(f"q = 2^{LOG_Q_32} PARALLEL SWEEP")
    print("=" * 70)

    sigma_values_32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]

    print("\n--- Parallel sigma sweep for n=1024, 2048 ---")
    sweep_sigma_for_n_parallel([1024, 2048], LOG_Q_32, sigma_values_32, "sweep_q32")

    # Other n values - parallel
    print("\n--- Testing other n values in parallel ---")
    sweep_sigma_for_n_parallel([512, 768, 1280, 1536], LOG_Q_32, [10.0, 50.0, 100.0], "other_q32")

    # q = 2^64 - parallel sweep
    print(f"\n{'='*70}")
    print(f"q = 2^{LOG_Q_64} PARALLEL SWEEP")
    print("=" * 70)

    sigma_values_64 = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                       500000.0, 1000000.0, 5000000.0, 10000000.0,
                       50000000.0, 100000000.0]

    print("\n--- Parallel sigma sweep for n=1024, 2048 ---")
    sweep_sigma_for_n_parallel([1024, 2048], LOG_Q_64, sigma_values_64, "sweep_q64")

    # Binary search for minimum n - uses parallel fine search
    print(f"\n{'='*70}")
    print("BINARY SEARCH FOR MINIMUM n (parallel fine search)")
    print("=" * 70)

    targets = [(2**16, 1024, 2048), (2**20, 1024, 3072), (2**24, 1500, 4096)]
    for target_p, n_start, n_end in targets:
        print(f"\n--- Target p = {target_p:,} (2^{int(math.log2(target_p))}) ---")
        best_n, sigma = find_min_n_binary_parallel(LOG_Q_64, target_p, n_start, n_end)
        if best_n:
            print(f"  RESULT: min n = {best_n} for p >= {target_p:,}")

    write_csv()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - BEST PARAMETERS")
    print("=" * 70)

    print("\nq=2^32 best results:")
    q32_pass = [r for r in all_results if r['log_q'] == 32 and r['passes'] == 'YES']
    for n in [1024, 2048]:
        n_results = [r for r in q32_pass if r['n'] == n]
        if n_results:
            best = min(n_results, key=lambda x: x['sigma'])
            print(f"  n={n}: sigma={best['sigma']:.2f}, p={best['p']:,} (2^{best['log2_p']:.1f}), security={best['security']:.1f}")

    print("\nq=2^64 best results:")
    q64_pass = [r for r in all_results if r['log_q'] == 64 and r['passes'] == 'YES']
    for n in [1024, 2048]:
        n_results = [r for r in q64_pass if r['n'] == n]
        if n_results:
            best = min(n_results, key=lambda x: x['sigma'])
            print(f"  n={n}: sigma={best['sigma']:.2f}, p={best['p']:,} (2^{best['log2_p']:.1f}), security={best['security']:.1f}")

if __name__ == "__main__":
    main()
