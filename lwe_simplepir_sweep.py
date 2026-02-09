#!/usr/bin/env python3
"""
LWE parameter sweep for SimplePIR.
Tests all sigma variants and finds optimal n using binary search.
Saves ALL intermediate results to CSV.
"""

import sys
import math
import csv
import os

# Add the repo root to path (assumes this script is run from repo root)
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

def get_security(n, log_q, sigma):
    from estimator import LWE, ND
    from sage.all import oo

    q = 2 ** log_q
    params = LWE.Parameters(
        n=n, q=q,
        Xs=ND.DiscreteGaussian(sigma),
        Xe=ND.DiscreteGaussian(sigma),
        m=m
    )
    result = LWE.estimate(params, jobs=1)

    min_bits = float("inf")
    for _, data in result.items():
        rop = data.get("rop", oo)
        if rop != oo:
            bits = math.log2(float(rop))
            if bits < min_bits:
                min_bits = bits
    return min_bits

def record_result(n, log_q, sigma, security, source):
    q = 2 ** log_q
    p = compute_p(q, sigma)
    log2_p = math.log2(p) if p > 0 else 0
    passes = security >= TARGET_SECURITY

    result = {
        'n': n,
        'log_q': log_q,
        'q': q,
        'sigma': sigma,
        'p': p,
        'log2_p': log2_p,
        'security': security,
        'passes': 'YES' if passes else 'NO',
        'source': source
    }
    all_results.append(result)

    status = "PASS" if passes else "FAIL"
    print(f"    n={n}, sigma={sigma:.2f}, p={p:,} (2^{log2_p:.1f}), security={security:.1f} [{status}]")
    return result

def sweep_sigma_for_n(n, log_q, sigma_values, source_prefix):
    print(f"\n  Testing n={n} with {len(sigma_values)} sigma values...")
    for sigma in sigma_values:
        sec = get_security(n, log_q, sigma)
        record_result(n, log_q, sigma, sec, f"{source_prefix}")

def find_min_n_binary(log_q, target_p, n_low, n_high):
    """Binary search to find minimum n for target p with 128-bit security."""
    q = 2 ** log_q
    sigma_needed = compute_sigma_for_p(q, target_p)

    print(f"\n  Binary search for min n with p>={target_p:,} (sigma={sigma_needed:.2f})")
    print(f"  Search range: [{n_low}, {n_high}]")

    best_n = None

    while n_high - n_low > 8:
        n_mid = (n_low + n_high) // 2
        n_mid = (n_mid // 8) * 8

        sec = get_security(n_mid, log_q, sigma_needed)
        record_result(n_mid, log_q, sigma_needed, sec, f"binary_search_p{target_p}")

        if sec >= TARGET_SECURITY:
            best_n = n_mid
            n_high = n_mid
        else:
            n_low = n_mid

    for n in range(n_low, n_high + 1, 8):
        sec = get_security(n, log_q, sigma_needed)
        record_result(n, log_q, sigma_needed, sec, f"fine_search_p{target_p}")
        if sec >= TARGET_SECURITY and (best_n is None or n < best_n):
            best_n = n

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
    print("SimplePIR LWE Parameter Sweep - ALL INTERMEDIATE RESULTS")
    print("=" * 70)
    print(f"Database size:    N = 2^{LOG_N} = {N:,}")
    print(f"LWE samples:      m = sqrt(N) = {m:,}")
    print(f"Correctness:      delta = 2^-40 = {DELTA:.2e}")
    print(f"Security target:  {TARGET_SECURITY} bits")
    print("=" * 70)

    # q = 2^32
    print(f"\n{'='*70}")
    print(f"q = 2^{LOG_Q_32} COMPREHENSIVE SWEEP")
    print("=" * 70)

    # Full sigma sweep for n=1024 and n=2048
    sigma_values_32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]

    print("\n--- Full sigma sweep for key n values ---")
    sweep_sigma_for_n(1024, LOG_Q_32, sigma_values_32, "sigma_sweep_q32")
    sweep_sigma_for_n(2048, LOG_Q_32, sigma_values_32, "sigma_sweep_q32")

    # Test other n values at sigma that might give 128-bit
    print("\n--- Testing other n values ---")
    for n in [512, 768, 1280, 1536]:
        for sigma in [10.0, 50.0, 100.0]:
            sec = get_security(n, LOG_Q_32, sigma)
            record_result(n, LOG_Q_32, sigma, sec, "other_n_q32")

    # q = 2^64
    print(f"\n{'='*70}")
    print(f"q = 2^{LOG_Q_64} COMPREHENSIVE SWEEP")
    print("=" * 70)

    # Full sigma sweep for n=1024 and n=2048
    sigma_values_64 = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                       500000.0, 1000000.0, 5000000.0, 10000000.0,
                       50000000.0, 100000000.0]

    print("\n--- Full sigma sweep for key n values ---")
    sweep_sigma_for_n(1024, LOG_Q_64, sigma_values_64, "sigma_sweep_q64")
    sweep_sigma_for_n(2048, LOG_Q_64, sigma_values_64, "sigma_sweep_q64")

    # Binary search to find minimum n for target p values
    print(f"\n{'='*70}")
    print("BINARY SEARCH FOR MINIMUM n")
    print("=" * 70)

    targets = [(2**16, 1024, 2048), (2**20, 1024, 3072), (2**24, 1500, 4096)]
    for target_p, n_start, n_end in targets:
        print(f"\n--- Target p = {target_p:,} (2^{int(math.log2(target_p))}) ---")
        best_n, sigma = find_min_n_binary(LOG_Q_64, target_p, n_start, n_end)
        if best_n:
            print(f"  RESULT: min n = {best_n} for p >= {target_p:,}")

    # Test around discovered boundaries
    print(f"\n{'='*70}")
    print("FINE-GRAINED SEARCH AROUND BOUNDARIES")
    print("=" * 70)

    # Around n=1226 (found as boundary for p=2^16)
    print("\n--- Around n=1226 boundary (q=2^64) ---")
    for n in range(1200, 1280, 16):
        sigma = compute_sigma_for_p(2**64, 2**16)
        sec = get_security(n, LOG_Q_64, sigma)
        record_result(n, LOG_Q_64, sigma, sec, "boundary_p65536")

    write_csv()

    # Print summary
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
