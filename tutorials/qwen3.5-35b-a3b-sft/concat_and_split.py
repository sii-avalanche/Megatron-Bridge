#!/usr/bin/env python3

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


try:
    import orjson

    def _dump_json_bytes(obj: Any) -> bytes:
        return orjson.dumps(obj) + b"\n"

    def _load_json(line: str) -> Any:
        return orjson.loads(line)
except ImportError:
    try:
        import ujson

        def _dump_json_bytes(obj: Any) -> bytes:
            return (ujson.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        def _load_json(line: str) -> Any:
            return ujson.loads(line)
    except ImportError:
        import json

        def _dump_json_bytes(obj: Any) -> bytes:
            return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        def _load_json(line: str) -> Any:
            return json.loads(line)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron-Cascade2 SFT data with Zero-IPC parallel processing."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500000)
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Optional JSON file receiving total, training, and validation record counts.",
    )
    return parser.parse_args()


def _iter_source_files(source_root: Path) -> list[Path]:
    files = []
    for path in sorted(source_root.rglob("*.jsonl")):
        rel_parts = path.relative_to(source_root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        files.append(path)
    return files


def get_byte_chunks(source_files: list[Path], chunk_size=64 * 1024 * 1024):
    """Compute byte-range chunks in the main process without reading full file contents."""
    chunks = []
    chunk_idx = 0
    for path in source_files:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            start = 0
            while start < file_size:
                seek_pos = start + chunk_size
                if seek_pos >= file_size:
                    end = file_size
                else:
                    f.seek(seek_pos)
                    f.readline()  # Advance to the next newline to avoid splitting lines.
                    end = f.tell()
                chunks.append((path, start, end, chunk_idx))
                start = end
                chunk_idx += 1
    return chunks


def _worker_process_chunk(args_tuple):
    path, start_byte, end_byte, chunk_idx, seed, val_ratio, tmp_dir = args_tuple

    train_tmp = Path(tmp_dir) / f"train_{chunk_idx}.jsonl"
    val_tmp = Path(tmp_dir) / f"val_{chunk_idx}.jsonl"

    domain_counter = Counter()
    train_count = val_count = total_count = 0

    hash_prefix = f"{seed}:".encode("utf-8")
    VAL_DIVISOR = 18446744073709551616.0

    with open(path, "rb") as f_in, open(train_tmp, "wb") as f_train, open(val_tmp, "wb") as f_val:
        f_in.seek(start_byte)

        while f_in.tell() < end_byte:
            raw_line_bytes = f_in.readline()
            if not raw_line_bytes:
                break

            raw_line = raw_line_bytes.decode("utf-8").strip()
            if not raw_line:
                continue

            try:
                obj = _load_json(raw_line)
            except Exception:
                continue

            if isinstance(obj, list):
                messages = obj
                domain = path.parent.name
            elif isinstance(obj, dict):
                messages = obj.get("messages")
                domain = obj.get("domain", path.parent.name)
            else:
                continue

            if not isinstance(messages, list):
                continue

            total_count += 1
            domain_counter[domain] += 1

            key = hash_prefix + raw_line_bytes
            digest = hashlib.blake2b(key, digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            is_val = (value / VAL_DIVISOR) < val_ratio

            out_bytes = _dump_json_bytes(obj)

            if is_val:
                f_val.write(out_bytes)
                val_count += 1
            else:
                f_train.write(out_bytes)
                train_count += 1

    return train_count, val_count, domain_counter, total_count


def main() -> None:
    """Split source JSONL records into deterministic training and validation files."""
    args = _parse_args()
    source_files = _iter_source_files(args.source_root)
    if not source_files:
        raise FileNotFoundError(f"No jsonl files found under: {args.source_root}")

    output_root = args.output_root
    train_path = output_root / "training.jsonl"
    valid_path = output_root / "validation.jsonl"
    tmp_dir = output_root / ".temp_chunks"

    output_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    auto_workers = max(1, (os.cpu_count() or 1) - 1)
    workers = auto_workers if args.workers == 0 else args.workers

    print(f"Parsing mode: Zero-IPC Disk Direct (workers={workers})")

    chunks = get_byte_chunks(source_files, chunk_size=64 * 1024 * 1024)
    total_chunks = len(chunks)

    task_args = [
        (path, start, end, chunk_idx, args.seed, args.val_ratio, str(tmp_dir))
        for path, start, end, chunk_idx in chunks
    ]

    train_count = val_count = total_count = 0
    domain_counter: Counter[str] = Counter()
    last_print = 0

    with mp.Pool(processes=workers) as pool:
        for t_count, v_count, d_counter, count in pool.imap_unordered(_worker_process_chunk, task_args):
            train_count += t_count
            val_count += v_count
            total_count += count
            domain_counter.update(d_counter)

            if args.progress_every > 0 and (total_count - last_print) >= args.progress_every:
                print(f"Progress: parsed={total_count}, train={train_count}, val={val_count}", flush=True)
                last_print = (total_count // args.progress_every) * args.progress_every

    print("\nAll workers finished! Merging temporary files via OS zero-copy...")
    with open(train_path, "wb") as f_train, open(valid_path, "wb") as f_val:
        for chunk_idx in range(total_chunks):
            t_tmp = tmp_dir / f"train_{chunk_idx}.jsonl"
            v_tmp = tmp_dir / f"val_{chunk_idx}.jsonl"

            if t_tmp.exists():
                with open(t_tmp, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_train)
                t_tmp.unlink()

            if v_tmp.exists():
                with open(v_tmp, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_val)
                v_tmp.unlink()

    tmp_dir.rmdir()

    if args.stats_path is not None:
        args.stats_path.parent.mkdir(parents=True, exist_ok=True)
        args.stats_path.write_text(
            json.dumps(
                {
                    "total_records": total_count,
                    "training_records": train_count,
                    "validation_records": val_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\nPrepared {total_count} records from {len(source_files)} files")
    print(f"Training samples:   {train_count} -> {train_path}")
    print(f"Validation samples: {val_count} -> {valid_path}")
    print("Domain distribution:")
    for domain, count in sorted(domain_counter.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
