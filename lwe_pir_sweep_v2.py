#!/usr/bin/env python3
"""
LWE Parameter Sweep for PIR - Version 2
Finds minimum n (precision=1) and minimum sigma (precision=0.1) for 128-bit security.
"""

import sys
import os
import math
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TARGET_SECURITY = 128
ESTIMATOR_JOBS = 4

def compute_p(q, sigma, N, delta):
    ln_term = math.log(2 / delta)
    denom = math.sqrt(2) * sigma * (N ** 0.25) * math.sqrt(ln_term)
    p_sq = q / denom
    return int(math.floor(math.sqrt(p_sq))) if p_sq > 0 else 0

def compute_sigma_for_p(q, target_p, N, delta):
    ln_term = math.log(2 / delta)
    sigma = q / (math.sqrt(2) * (target_p ** 2) * (N ** 0.25) * math.sqrt(ln_term))
    return sigma

def get_security(n, log_q, sigma, N, m, delta, verbose=True):
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

        p = compute_p(q, sigma, N, delta)
        passes = min_bits >= TARGET_SECURITY

        if verbose:
            status = "PASS" if passes else "FAIL"
            print(f"    n={n}, sigma={sigma:.2f} -> p={p:,}, security={min_bits:.1f} [{status}]")

        return {
            'n': n, 'log_q': log_q, 'sigma': sigma,
            'p': p, 'log2_p': math.log2(p) if p > 0 else 0,
            'security': min_bits,
            'passes': passes
        }
    except Exception as e:
        if verbose:
            print(f"    n={n}, sigma={sigma:.2f} -> ERROR: {e}")
        return {'n': n, 'log_q': log_q, 'sigma': sigma, 'p': 0,
                'log2_p': 0, 'security': 0, 'passes': False, 'error': str(e)}

def find_min_n_for_sigma(log_q, sigma, N, m, delta, n_low, n_high):
    """Binary search to find minimum n for given sigma with 128-bit security (precision=1)."""
    print(f"  Finding min n for sigma={sigma:.2f} in range [{n_low}, {n_high}]...")

    best_n = None
    best_result = None

    # Binary search with precision 1
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        result = get_security(n_mid, log_q, sigma, N, m, delta)

        if result['passes']:
            best_n = n_mid
            best_result = result
            n_high = n_mid
        else:
            n_low = n_mid

    # Check the boundary
    if best_n is None or n_low + 1 < best_n:
        result = get_security(n_low + 1, log_q, sigma, N, m, delta)
        if result['passes']:
            best_n = n_low + 1
            best_result = result

    return best_n, best_result

def find_min_sigma_for_n(n, log_q, N, m, delta, min_p, sigma_low, sigma_high):
    """Binary search to find minimum sigma for given n with 128-bit security (precision=0.1)."""
    print(f"  Finding min sigma for n={n} in range [{sigma_low:.2f}, {sigma_high:.2f}]...")

    best_sigma = None
    best_result = None

    # Binary search with precision 0.1
    while sigma_high - sigma_low > 0.1:
        sigma_mid = (sigma_low + sigma_high) / 2
        result = get_security(n, log_q, sigma_mid, N, m, delta)

        if result['passes'] and result['p'] >= min_p:
            best_sigma = sigma_mid
            best_result = result
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    return best_sigma, best_result

def find_optimal_params(log_q, target_p, N, m, delta, n_range=(1024, 4096)):
    """Find minimum (n, sigma) pair for target p with 128-bit security."""
    q = 2 ** log_q
    sigma = compute_sigma_for_p(q, target_p, N, delta)

    print(f"\n--- Target p >= {target_p:,} (2^{math.log2(target_p):.0f}), required sigma = {sigma:.2f} ---")

    # Step 1: Find minimum n for this sigma
    min_n, result = find_min_n_for_sigma(log_q, sigma, N, m, delta, n_range[0], n_range[1])

    if min_n is None:
        print(f"  No valid n found in range {n_range}")
        return None

    print(f"  Found min n = {min_n} for sigma={sigma:.2f}")

    # Step 2: For this minimum n, find the minimum sigma that still gives 128-bit security
    # Start from a lower sigma and search upward
    sigma_low = sigma * 0.5
    sigma_high = sigma * 2

    min_sigma, final_result = find_min_sigma_for_n(min_n, log_q, N, m, delta, target_p, sigma_low, sigma_high)

    if min_sigma:
        print(f"  OPTIMAL: n={min_n}, sigma={min_sigma:.2f}, p={final_result['p']:,}, security={final_result['security']:.1f}")
        return final_result
    else:
        return result

def main():
    parser = argparse.ArgumentParser(description='LWE Parameter Sweep v2 - Find minimum (n, sigma) pairs')
    parser.add_argument('--log_N', type=int, required=True,
                        help='log2(N) database size (e.g., 36, 37, 38, 39)')
    parser.add_argument('--log_q', type=int, required=True, choices=[32, 64],
                        help='log2(q) ciphertext modulus: 32 or 64')
    parser.add_argument('--log_delta', type=int, default=40,
                        help='log2(1/delta) correctness param (default: 40, i.e., delta=2^-40)')
    parser.add_argument('--target_p', type=str, default=None,
                        help='Comma-separated target p values as powers of 2 (e.g., "16,20,24" for 2^16, 2^20, 2^24)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file')
    parser.add_argument('--n_min', type=int, default=1024,
                        help='Minimum n to search (default: 1024)')
    parser.add_argument('--n_max', type=int, default=4096,
                        help='Maximum n to search (default: 4096)')

    args = parser.parse_args()

    log_N = args.log_N
    N = 2 ** log_N
    m = int(math.ceil(math.sqrt(N)))
    log_q = args.log_q
    q = 2 ** log_q
    log_delta = args.log_delta
    delta = 2 ** (-log_delta)
    n_range = (args.n_min, args.n_max)

    # Default target p values
    if args.target_p:
        target_p_powers = [int(x.strip()) for x in args.target_p.split(',')]
    else:
        if log_q == 32:
            target_p_powers = [8, 9, 10]  # 2^8=256, 2^9=512, 2^10=1024
        else:
            target_p_powers = [16, 18, 20, 22, 24]  # 2^16 to 2^24

    output_file = args.output or f"lwe_optimal_N{log_N}_q{log_q}_d{log_delta}.csv"

    print("=" * 70)
    print("LWE Parameter Sweep v2 - Finding Minimum (n, sigma) Pairs")
    print("=" * 70)
    print(f"Database:     N = 2^{log_N} = {N:,}")
    print(f"Samples:      m = sqrt(N) = {m:,}")
    print(f"Ciphertext:   q = 2^{log_q}")
    print(f"Correctness:  delta = 2^-{log_delta}")
    print(f"n range:      [{n_range[0]}, {n_range[1]}]")
    print(f"Target p:     2^{target_p_powers}")
    print(f"Output:       {output_file}")
    print("=" * 70)

    all_results = []

    for p_power in target_p_powers:
        target_p = 2 ** p_power
        result = find_optimal_params(log_q, target_p, N, m, delta, n_range)
        if result:
            all_results.append({
                'log_N': log_N, 'N': N, 'm': m,
                'n': result['n'], 'log_q': log_q, 'q': q,
                'log_delta': log_delta,
                'sigma': result['sigma'],
                'p': result['p'], 'log2_p': result['log2_p'],
                'target_log2_p': p_power,
                'security': result['security']
            })

    # Write CSV
    if all_results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['log_N', 'N', 'm', 'n', 'log_q', 'q', 'log_delta', 'sigma', 'p', 'log2_p', 'target_log2_p', 'security_bits'])
            for r in all_results:
                writer.writerow([
                    r['log_N'], r['N'], r['m'], r['n'], r['log_q'], r['q'],
                    r['log_delta'], f"{r['sigma']:.2f}", r['p'], f"{r['log2_p']:.1f}",
                    r['target_log2_p'], f"{r['security']:.1f}"
                ])
        print(f"\n{len(all_results)} results saved to {output_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"OPTIMAL RESULTS: N=2^{log_N}, q=2^{log_q}, delta=2^-{log_delta}")
    print("=" * 80)
    print(f"{'target_p':>10} | {'n':>6} | {'sigma':>12} | {'actual_p':>12} | {'log2(p)':>8} | {'security':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{'2^'+str(r['target_log2_p']):>10} | {r['n']:>6} | {r['sigma']:>12.2f} | {r['p']:>12,} | {r['log2_p']:>8.1f} | {r['security']:>10.1f}")

if __name__ == "__main__":
    main()
