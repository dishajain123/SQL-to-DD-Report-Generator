"""Reproducible offline performance and completeness check for one SQL file.

python -m scripts.benchmark_large_sql --input procedure.sql --output-dir output/benchmark
Exit 0 = complete coverage, 1 = failed run, 2 = review required.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from app.batch import _configure_storage, run_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    _configure_storage(args.output_dir)
    started = time.perf_counter()
    result = run_file(args.input.name, data)
    elapsed = time.perf_counter() - started
    path = Path(result.output_dir) / 'completeness.json'
    completeness = json.loads(path.read_text()) if path.exists() else {}
    report = {
        'input_bytes': len(data),
        'elapsed_seconds': round(elapsed, 3),
        'ready': bool(completeness.get('ready')),
        'result': asdict(result),
    }
    metrics = args.output_dir / 'benchmark.json'
    metrics.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{elapsed:.3f}s; {result.rows} rows; {result.active} active; '
          f'{len(result.parse_failures)} parse failures; ready={report["ready"]}')
    print(f'Metrics: {metrics}\nArtifacts: {result.output_dir}')
    return 1 if result.error else (0 if report['ready'] else 2)


if __name__ == '__main__':
    raise SystemExit(main())
