#!/usr/bin/env python3
"""Concatenate valid source JSONL records into one training JSONL file."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


try:
    import orjson

    def _load_json(line: str) -> Any:
        return orjson.loads(line)

    def _dump_json(obj: Any) -> bytes:
        return orjson.dumps(obj) + b"\n"
except ImportError:
    def _load_json(line: str) -> Any:
        return json.loads(line)

    def _dump_json(obj: Any) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500000)
    parser.add_argument("--stats-path", type=Path)
    return parser.parse_args()


def _iter_source_files(source_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(source_root.rglob("*.jsonl"))
        if not any(part.startswith(".") for part in path.relative_to(source_root).parts)
    ]


def _worker(args: tuple[Path, Path]) -> tuple[Path, Counter[str], int]:
    source, temporary = args
    domains: Counter[str] = Counter()
    total = 0
    with source.open("rb") as source_file, temporary.open("wb") as output_file:
        for raw in source_file:
            if not raw.strip():
                continue
            try:
                obj = _load_json(raw.decode("utf-8").strip())
            except Exception:
                continue
            if isinstance(obj, list):
                messages, domain = obj, source.parent.name
            elif isinstance(obj, dict):
                messages, domain = obj.get("messages"), obj.get("domain", source.parent.name)
            else:
                continue
            if not isinstance(messages, list):
                continue
            output_file.write(_dump_json(obj))
            domains[str(domain)] += 1
            total += 1
    return temporary, domains, total


def main() -> None:
    args = _parse_args()
    source_files = _iter_source_files(args.source_root)
    if not source_files:
        raise FileNotFoundError(f"No jsonl files found under: {args.source_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = args.output_root / ".concat_tmp"
    temporary_root.mkdir(exist_ok=True)
    workers = max(1, (os.cpu_count() or 1) - 1) if args.workers == 0 else args.workers
    tasks = [(source, temporary_root / f"part_{index:06d}.jsonl")
             for index, source in enumerate(source_files)]
    output = args.output_root / "training.jsonl"
    total = 0
    domains: Counter[str] = Counter()
    try:
        with output.open("wb") as destination:
            with mp.Pool(processes=workers) as pool:
                for temporary, counts, count in pool.imap(_worker, tasks):
                    with temporary.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                    temporary.unlink()
                    total += count
                    domains.update(counts)
                    if args.progress_every > 0 and total % args.progress_every < count:
                        print(f"Progress: parsed={total}, training={total}", flush=True)
    finally:
        if temporary_root.exists():
            for temporary in temporary_root.glob("*.jsonl"):
                temporary.unlink()
            temporary_root.rmdir()
    if args.stats_path:
        args.stats_path.parent.mkdir(parents=True, exist_ok=True)
        args.stats_path.write_text(
            json.dumps({"total_records": total, "training_records": total}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Prepared {total} training records from {len(source_files)} files")
    print(f"Training samples: {total} -> {output}")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
