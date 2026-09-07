#!/usr/bin/env python3
"""Split one JSONL into N randomly assigned JSONL parts with streaming I/O.

This combines the logical effect of:
1) shuffling records
2) splitting into N files

without loading the full file into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing as mp
import os
import random
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)


def _compute_byte_chunks(path: Path, chunk_size: int) -> list[tuple[int, int, int]]:
    size = path.stat().st_size
    if size == 0:
        return []

    chunks: list[tuple[int, int, int]] = []
    chunk_idx = 0
    with path.open("rb") as f:
        start = 0
        while start < size:
            seek_pos = start + chunk_size
            if seek_pos >= size:
                end = size
            else:
                f.seek(seek_pos)
                f.readline()
                end = f.tell()
            chunks.append((start, end, chunk_idx))
            start = end
            chunk_idx += 1
    return chunks


def _worker_process_chunk(
    args_tuple: tuple[Path, int, int, int, int, int, Path],
) -> tuple[int, list[int]]:
    src_path, start_byte, end_byte, chunk_idx, num_parts, seed, tmp_dir = args_tuple
    out_paths = [tmp_dir / f"part_{i:05d}_chunk_{chunk_idx:06d}.jsonl" for i in range(num_parts)]
    handles = [p.open("wb") for p in out_paths]

    written_per_part = [0] * num_parts
    total = 0
    seed_prefix = f"{seed}:".encode("utf-8")

    try:
        with src_path.open("rb") as f_in:
            f_in.seek(start_byte)
            while f_in.tell() < end_byte:
                line = f_in.readline()
                if not line:
                    break
                if not line.strip():
                    continue

                digest = hashlib.blake2b(seed_prefix + line, digest_size=8).digest()
                part_idx = int.from_bytes(digest, byteorder="big", signed=False) % num_parts
                handles[part_idx].write(line)
                written_per_part[part_idx] += 1
                total += 1
    finally:
        for h in handles:
            h.close()

    return total, written_per_part


def _shuffle_file_lines(path: Path, *, seed: int, part_idx: int) -> None:
    """Shuffle all lines in a JSONL file deterministically."""
    with path.open("rb") as f_in:
        lines = f_in.readlines()
    rnd = random.Random(f"{seed}:{part_idx}:line_shuffle")
    rnd.shuffle(lines)
    with path.open("wb") as f_out:
        f_out.writelines(lines)


def _merge_one_part(
    args_tuple: tuple[Path, Path, str, int, int, bool, int],
) -> int:
    """Merge all temporary chunk files for one output part and optionally shuffle."""
    output_dir, tmp_dir, output_stem, part_idx, num_chunks, shuffle_final_merge, seed = args_tuple
    dst = output_dir / f"{output_stem}_{part_idx:05d}.jsonl"
    with dst.open("wb") as f_out:
        for chunk_idx in range(num_chunks):
            src_part = tmp_dir / f"part_{part_idx:05d}_chunk_{chunk_idx:06d}.jsonl"
            if src_part.exists():
                with src_part.open("rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                src_part.unlink()
    if shuffle_final_merge and dst.exists():
        _shuffle_file_lines(dst, seed=seed, part_idx=part_idx)
    return part_idx


def parse_args() -> argparse.Namespace:
    """Parse options for deterministic JSONL shuffling and splitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory for split files.")
    parser.add_argument("--num-parts", "-n", type=int, required=True, help="Number of output JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic random assignment.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes; 0 means max(1, cpu_count-1).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=64,
        help="Input byte chunk size (MiB) for parallel processing.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="part",
        help="Output basename stem: <stem>_00000.jsonl, <stem>_00001.jsonl, ...",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Progress report frequency by processed line count; <=0 disables.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing <output-stem>_*.jsonl files in output-dir before writing.",
    )
    parser.add_argument(
        "--shuffle-final-merge",
        action="store_true",
        help="Shuffle all lines in each final output file (deterministic with --seed).",
    )
    return parser.parse_args()


def main() -> None:
    """Split the input JSONL into shuffled output chunks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.num_parts < 1:
        raise ValueError("--num-parts must be >= 1")
    if args.chunk_size_mb < 1:
        raise ValueError("--chunk-size-mb must be >= 1")

    src = args.input.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input is not a file: {src}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for stale in output_dir.glob(f"{args.output_stem}_*.jsonl"):
            if stale.is_file():
                stale.unlink()

    tmp_dir = output_dir / ".split_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, (os.cpu_count() or 1) - 1) if args.workers == 0 else args.workers
    chunk_size_bytes = args.chunk_size_mb * 1024 * 1024
    chunks = _compute_byte_chunks(src, chunk_size_bytes)
    if not chunks:
        raise ValueError(f"Input file is empty: {src}")

    logger.info(
        "Split+shuffle %s into %d parts (workers=%d, chunks=%d)",
        src,
        args.num_parts,
        workers,
        len(chunks),
    )

    task_args = [(src, start, end, chunk_idx, args.num_parts, args.seed, tmp_dir) for start, end, chunk_idx in chunks]

    processed = 0
    written_per_part = [0] * args.num_parts
    last_report = 0

    with mp.Pool(processes=workers) as pool:
        for count, part_counts in pool.imap_unordered(_worker_process_chunk, task_args):
            processed += count
            for i, n in enumerate(part_counts):
                written_per_part[i] += n

            if args.progress_every > 0 and (processed - last_report) >= args.progress_every:
                logger.info("Progress: processed=%d", processed)
                last_report = (processed // args.progress_every) * args.progress_every

    logger.info("Merging temporary files into final outputs...")
    merge_workers = min(workers, args.num_parts)
    merge_task_args = [
        (
            output_dir,
            tmp_dir,
            args.output_stem,
            i,
            len(chunks),
            args.shuffle_final_merge,
            args.seed,
        )
        for i in range(args.num_parts)
    ]
    with mp.Pool(processes=merge_workers) as pool:
        for _ in pool.imap_unordered(_merge_one_part, merge_task_args):
            pass

    tmp_dir.rmdir()

    logger.info("Done. Total records written: %d", processed)
    non_empty = 0
    for i in range(args.num_parts):
        dst = output_dir / f"{args.output_stem}_{i:05d}.jsonl"
        c = written_per_part[i]
        if c > 0:
            non_empty += 1
        logger.info("  %s: %d records", dst.name, c)
    logger.info("Non-empty output files: %d / %d", non_empty, args.num_parts)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
