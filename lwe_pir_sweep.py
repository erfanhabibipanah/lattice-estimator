#!/usr/bin/env python3
"""
LWE Parameter Sweep for PIR - Optimized CLI Version
Only outputs parameters that achieve 128-bit security.
"""

import sys
import os
import math
import csv
import argparse
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_LOG_DELTA = 40
TARGET_SECURITY = 128
NUM_WORKERS = max(1, cpu_count() - 1)
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

def get_security(args):
    n, log_q, sigma, N, m, delta = args
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
        return {
            'n': n, 'log_q': log_q, 'sigma': sigma,
            'p': p, 'log2_p': math.log2(p) if p > 0 else 0,
            'security': min_bits,
            'passes': min_bits >= TARGET_SECURITY
        }
    except Exception as e:
        return {'n': n, 'log_q': log_q, 'sigma': sigma, 'p': 0,
                'log2_p': 0, 'security': 0, 'passes': False, 'error': str(e)}

def parallel_estimate(tasks, num_workers=None):
    """Run estimates - uses sequential for reliability with Sage."""
    # Multiprocessing with Sage can have issues, use sequential
    results = []
    for i, task in enumerate(tasks):
        print(f"    [{i+1}/{len(tasks)}] n={task[0]}, sigma={task[2]:.2f}...", end=" ", flush=True)
        r = get_security(task)
        status = "PASS" if r['passes'] else "FAIL"
        print(f"p={r['p']:,}, security={r['security']:.1f} [{status}]")
        results.append(r)
    return results

def generate_sigma_range(log_q, target_p, N, delta):
    """Generate smart sigma range based on parameters."""
    q = 2 ** log_q
    sigma_max = compute_sigma_for_p(q, target_p, N, delta) * 2

    if log_q == 32:
        # Coarse range + fine steps around the 128-bit threshold (10-15)
        sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0, 15.0,
                  20.0, 30.0, 50.0, 100.0]
    else:
        sigmas = []
        s = 1.0
        while s < min(sigma_max * 10, 2 ** log_q):
            sigmas.append(s)
            s *= 10
        sigmas.extend([s * 0.5 for s in sigmas if s * 0.5 not in sigmas])
        sigmas = sorted(set(sigmas))

    return [s for s in sigmas if s <= 2 ** log_q]

def find_min_n_for_p(log_q, target_p, N, m, n_low, n_high, delta):
    """Binary search to find minimum n for target p with 128-bit security."""
    q = 2 ** log_q
    sigma = compute_sigma_for_p(q, target_p, N, delta)

    print(f"  Searching min n for p >= {target_p:,} (sigma = {sigma:.2f})")

    best_n = None
    best_result = None

    while n_high - n_low > 32:
        n_mid = ((n_low + n_high) // 2 // 8) * 8
        result = get_security((n_mid, log_q, sigma, N, m, delta))

        if result['passes']:
            best_n = n_mid
            best_result = result
            n_high = n_mid
        else:
            n_low = n_mid

    # Fine search in parallel
    fine_tasks = [(n, log_q, sigma, N, m, delta) for n in range(n_low, n_high + 1, 8)]
    if fine_tasks:
        fine_results = parallel_estimate(fine_tasks)
        for r in fine_results:
            if r['passes'] and (best_n is None or r['n'] < best_n):
                best_n = r['n']
                best_result = r

    return best_n, best_result

def sweep_for_best_sigma(n, log_q, N, m, min_p, delta):
    """Find ALL sigma values that give 128-bit security for given n.

    Returns: (all_passing_results, best_result)
    """
    q = 2 ** log_q
    sigmas = generate_sigma_range(log_q, min_p, N, delta)

    tasks = [(n, log_q, s, N, m, delta) for s in sigmas]
    results = parallel_estimate(tasks)

    passing = [r for r in results if r['passes'] and r['p'] >= min_p]
    failing = [r for r in results if not r['passes'] and r['p'] >= min_p]

    if not passing:
        return [], None

    best = min(passing, key=lambda x: x['sigma'])

    # Binary search for exact minimum sigma between last fail and first pass
    if failing:
        last_fail = max(failing, key=lambda x: x['sigma'])
        if last_fail['sigma'] < best['sigma']:
            print(f"  Refining between sigma={last_fail['sigma']:.2f} and {best['sigma']:.2f}...")
            sigma_low = last_fail['sigma']
            sigma_high = best['sigma']

            # Binary search with 0.1 precision
            while sigma_high - sigma_low > 0.1:
                sigma_mid = (sigma_low + sigma_high) / 2
                print(f"    Testing sigma={sigma_mid:.2f}...", end=" ", flush=True)
                r = get_security((n, log_q, sigma_mid, N, m, delta))
                status = "PASS" if r['passes'] else "FAIL"
                print(f"p={r['p']:,}, security={r['security']:.1f} [{status}]")

                if r['passes'] and r['p'] >= min_p:
                    best = r
                    passing.append(r)  # Add refined result to passing list
                    sigma_high = sigma_mid
                else:
                    sigma_low = sigma_mid

    return passing, best

def main():
    parser = argparse.ArgumentParser(description='LWE Parameter Sweep for PIR')
    parser.add_argument('--log_N', type=int, required=True, choices=[36, 37, 38, 39],
                        help='log2(N) database size: 36, 37, 38, or 39')
    parser.add_argument('--log_q', type=int, required=True, choices=[32, 64],
                        help='log2(q) ciphertext modulus: 32 or 64')
    parser.add_argument('--log_delta', type=int, default=DEFAULT_LOG_DELTA,
                        help=f'log2(1/delta) correctness param (default: {DEFAULT_LOG_DELTA}, i.e., delta=2^-40)')
    parser.add_argument('--min_p', type=int, default=None,
                        help='Minimum p value (default: 256 for q=32, 2^16 for q=64)')
    parser.add_argument('--n_values', type=str, default=None,
                        help='Comma-separated n values to test (e.g., "1024,1536,2048")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (default: lwe_results_N{log_N}_q{log_q}.csv)')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel workers (default: cpu_count - 1)')

    args = parser.parse_args()

    num_workers = args.workers if args.workers else NUM_WORKERS

    log_N = args.log_N
    N = 2 ** log_N
    m = int(math.ceil(math.sqrt(N)))
    log_q = args.log_q
    q = 2 ** log_q
    log_delta = args.log_delta
    delta = 2 ** (-log_delta)

    if args.min_p is None:
        min_p = 256 if log_q == 32 else 2 ** 16
    else:
        min_p = args.min_p

    if args.n_values:
        n_values = [int(x.strip()) for x in args.n_values.split(',')]
    else:
        if log_q == 32:
            n_values = [1024, 1280, 1536, 2048]
        else:
            n_values = [1024, 1224, 1536, 1696, 2048, 2248]

    output_file = args.output or f"lwe_results_N{log_N}_q{log_q}.csv"

    print("=" * 70)
    print("LWE Parameter Sweep for PIR")
    print("=" * 70)
    print(f"Database:     N = 2^{log_N} = {N:,}")
    print(f"Samples:      m = sqrt(N) = {m:,}")
    print(f"Ciphertext:   q = 2^{log_q}")
    print(f"Correctness:  delta = 2^-{log_delta}")
    print(f"Min p:        {min_p:,} (2^{math.log2(min_p):.1f})")
    print(f"n values:     {n_values}")
    print(f"Workers:      {num_workers}")
    print(f"Output:       {output_file}")
    print("=" * 70)

    all_results = []
    best_results = {}  # Track best result per n

    # Sweep each n value
    print(f"\n--- Finding best sigma for each n (p >= {min_p:,}) ---\n")
    for n in n_values:
        print(f"Testing n = {n}:")
        passing_results, best = sweep_for_best_sigma(n, log_q, N, m, min_p, delta)
        if passing_results:
            # Store best for this n
            best_results[n] = best['sigma']

            # Add ALL passing results
            for r in passing_results:
                all_results.append({
                    'log_N': log_N, 'N': N, 'm': m,
                    'n': r['n'], 'log_q': log_q, 'q': q,
                    'sigma': r['sigma'],
                    'p': r['p'], 'log2_p': r['log2_p'],
                    'security': r['security'],
                    'log_delta': log_delta
                })
            print(f"  BEST: sigma={best['sigma']:.2f}, p={best['p']:,} (2^{best['log2_p']:.1f}), security={best['security']:.1f} bits")
            print(f"  Total passing: {len(passing_results)} configurations\n")
        else:
            print(f"  No valid parameters found for p >= {min_p:,}\n")

    # Find minimum n for specific p targets
    if log_q == 64:
        print(f"\n--- Finding minimum n for target p values ---\n")
        targets = [2**16, 2**20, 2**24]
        for target_p in targets:
            n_low = 1024
            n_high = 4096 if target_p >= 2**24 else 2048

            best_n, result = find_min_n_for_p(log_q, target_p, N, m, n_low, n_high, delta)
            if best_n and result:
                all_results.append({
                    'log_N': log_N, 'N': N, 'm': m,
                    'n': result['n'], 'log_q': log_q, 'q': q,
                    'sigma': result['sigma'],
                    'p': result['p'], 'log2_p': result['log2_p'],
                    'security': result['security'],
                    'log_delta': log_delta
                })
                print(f"  p >= 2^{int(math.log2(target_p))}: min n = {best_n}, sigma = {result['sigma']:.2f}, security = {result['security']:.1f} bits")

    # Write CSV (all passing results with BEST marker)
    if all_results:
        # Sort by n, then by sigma
        all_results_sorted = sorted(all_results, key=lambda x: (x['n'], x['sigma']))

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['log_N', 'N', 'm', 'n', 'log_q', 'q', 'log_delta', 'sigma', 'p', 'log2_p', 'security_bits', 'BEST'])
            for r in all_results_sorted:
                # Mark as BEST if this is the minimum sigma for this n
                is_best = "YES" if r['n'] in best_results and abs(r['sigma'] - best_results[r['n']]) < 0.01 else ""
                writer.writerow([
                    r['log_N'], r['N'], r['m'], r['n'], r['log_q'], r['q'],
                    r['log_delta'], f"{r['sigma']:.2f}", r['p'], f"{r['log2_p']:.1f}", f"{r['security']:.1f}", is_best
                ])
        print(f"\n{len(all_results)} passing results saved to {output_file}")

    # Print summary table (only best results)
    print(f"\n{'='*70}")
    print("BEST RESULTS (128-bit security, minimum sigma per n)")
    print("=" * 70)
    print(f"{'n':>6} | {'sigma':>12} | {'p':>15} | {'log2(p)':>8} | {'security':>10}")
    print("-" * 60)
    # Only show best results (one per n)
    seen_n = set()
    for r in sorted(all_results, key=lambda x: (x['n'], x['sigma'])):
        if r['n'] not in seen_n:
            seen_n.add(r['n'])
            print(f"{r['n']:>6} | {r['sigma']:>12.2f} | {r['p']:>15,} | {r['log2_p']:>8.1f} | {r['security']:>10.1f}")

    print(f"\nTotal passing configurations: {len(all_results)}")
    print(f"CSV file contains ALL passing (n, sigma) combinations with BEST column.")

if __name__ == "__main__":
    main()
