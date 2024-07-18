"""TODO this needs to be compatible with OSCAR's implementation

https://oscar-project.github.io/documentation/versions/oscar-2301/#locality-sentitive-hashing
"""

import datetime
import json
import logging
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path

import polars as pl
from smart_open import open
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def exact_dedup(
    input_dir,
    output_dir,
    output_gzip=False,
    override=False,
    workers=None,
    max_lines_per_file=0,
    max_files=0,
    print_file_progress=False,
    compute_hashes=False,
    hash_key="tlsh",
    glob_pattern="*.jsonl.gz",  # TODO only JSONL is supported
):
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    output_dir_path = Path(output_dir)

    input_dir_path = Path(input_dir)

    start_time = datetime.datetime.now()

    if not output_dir_path.exists():
        logger.warning(f"Output directory did not exist but was created: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

    def compute_hash_and_get_position(file_path):
        raise NotImplementedError

    def get_existing_hash_and_position(file_path):
        logger.debug(f"Reading from {file_path}")
        rows = []

        with open(file_path, "r") as file_handler:
            line_iter = enumerate(file_handler)

            if max_lines_per_file and print_file_progress:
                line_iter = tqdm(line_iter, total=max_lines_per_file)

            for line_idx, line in line_iter:
                doc = json.loads(line)

                if doc[hash_key]:
                    rows.append((line_idx, doc[hash_key]))
                # else:
                #     pass

                if max_lines_per_file > 0 and line_idx >= max_lines_per_file:
                    break

        return file_path, rows

    logger.info(f"Workers: {workers} (available cpu cores: {multiprocessing.cpu_count()})")

    all_file_paths = set(input_dir_path.rglob(glob_pattern))

    logger.info(f"Files found in input dir: {len(all_file_paths)}")

    # sort for reproducibility
    all_file_paths = list(sorted(all_file_paths))

    if max_files > 0:
        logger.info(f"Limiting files from {len(all_file_paths)} to {max_files}")

        all_file_paths = all_file_paths[:max_files]

    nice_all_file_paths = [str(p.relative_to(input_dir_path)) for p in all_file_paths]

    logger.info(f"Final input files: {nice_all_file_paths[:10]} (total: {len(all_file_paths)})")

    file_paths_temp_path = output_dir_path / "temp.file_paths.json"
    hash_temp_path = output_dir_path / "temp.hashes.csv"
    dedup_hash_temp_path = output_dir_path / "temp.dedup_hashes.csv"

    # Step 1: save file paths
    if not file_paths_temp_path.exists():
        # save temp files
        json.dump(nice_all_file_paths, open(file_paths_temp_path, "w"), indent=4)

        logger.info(f"File paths saved to {file_paths_temp_path}")
    else:
        logger.warning(f"File paths exist already at {file_paths_temp_path}")

        file_paths_from_disk = json.load(open(file_paths_temp_path))

        assert nice_all_file_paths == file_paths_from_disk

    # Step 2: extract hash values to CSV file
    if not hash_temp_path.exists():
        logger.info("Extracting hash values")

        with open(hash_temp_path, "w") as out_f:
            # header
            out_f.write("file_idx,line_idx,hash\n")

            with multiprocessing.Pool(workers) as pool:
                if compute_hashes:
                    worker_func = compute_hash_and_get_position
                else:
                    worker_func = get_existing_hash_and_position

                out_iter = pool.imap_unordered(worker_func, all_file_paths)

                # TODO build CSV output already in worker function!
                for i, (file_path, rows) in enumerate(out_iter, 1):
                    nice_path = str(file_path.relative_to(input_dir_path))
                    file_idx = str(nice_all_file_paths.index(nice_path))
                    for line_idx, hash in rows:
                        hash = hash[5:]  # strip "tlsh:"
                        out_f.write(file_idx + "," + str(line_idx) + "," + hash + "\n")

                    logger.info(f"{i}/{len(all_file_paths)} done")

    else:
        logger.warning(f"Hash values exist already at {hash_temp_path}")

    # Step 3: dedup hashes
    if (file_paths_temp_path.exists() and hash_temp_path.exists()) and not dedup_hash_temp_path.exists():
        logger.info(f"Dedup extracted hashes from {hash_temp_path} ...")
        hashes_df = pl.read_csv(hash_temp_path)
        logger.info(f"Loaded {len(hashes_df)} hashes ...")
        hashes_df.unique(
            "hash",
            keep="any",
        ).write_csv(dedup_hash_temp_path)

        logger.info(f"Dedup hashes saved to {dedup_hash_temp_path}")
    else:
        logger.warning(f"Dedup hashes exist already at {dedup_hash_temp_path}")

    # Step 4: copy dedup docs
    if dedup_hash_temp_path.exists():

        def selective_copy(worker_args):
            relative_file_path, selected_line_indicies = worker_args

            input_file_path = input_dir_path / relative_file_path
            output_file_path = output_dir_path / relative_file_path
            selected_line_indicies = set(selected_line_indicies)

            if not output_file_path.parent.exists():
                os.makedirs(output_file_path.parent, exist_ok=True)

            copied_lines = 0
            skipped_lines = 0

            if output_file_path.exists() and not override:
                logger.error(f"Cannot copy, output exists already at: {output_file_path}")

            else:
                with open(output_file_path, "w") as out_f:
                    with open(input_file_path, "r") as in_f:
                        for line_idx, line in enumerate(in_f):
                            if line_idx in selected_line_indicies:
                                out_f.write(line)
                                copied_lines += 1
                            else:
                                skipped_lines += 1

                            if max_lines_per_file > 0 and line_idx >= max_lines_per_file:
                                break

                logger.info(f"Copied {copied_lines:,} lines from {str(input_file_path)} to {str(output_file_path)}")

            return relative_file_path, copied_lines, skipped_lines

        # select from jsonl
        dedup_hashes_df = pl.read_csv(dedup_hash_temp_path).drop("hash")
        file_path_to_line_idxs = defaultdict(list)

        for file_idx, line_idx in dedup_hashes_df.iter_rows():
            file_path_to_line_idxs[nice_all_file_paths[file_idx]].append(int(line_idx))

        logger.info(f"Selecting {len(dedup_hashes_df):,} lines from {len(file_path_to_line_idxs):,} files")

        file_stats = []
        total_skipped_lines = 0
        total_copied_lines = 0

        with multiprocessing.Pool(workers) as pool:
            for i, (relative_file_path, copied_lines, skipped_lines) in enumerate(
                pool.imap_unordered(selective_copy, file_path_to_line_idxs.items())
            ):
                total_skipped_lines += skipped_lines
                total_copied_lines += copied_lines

                file_stats.append(
                    {
                        "input_file_path": str(input_dir_path / relative_file_path),
                        "output_file_path": str(output_dir_path / relative_file_path),
                        "skipped_lines": skipped_lines,
                        "copied_lines": copied_lines,
                    }
                )

                logger.info(f"done {i}/{len(file_path_to_line_idxs)}")

        logger.info(f"All files copied. {total_copied_lines=}; {total_skipped_lines=}")

        end_time = datetime.datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOBID", 0)

        stats = {
            "files": file_stats,
            "total_copied_lines": total_copied_lines,
            "total_skipped_lines": total_skipped_lines,
            "start_time": start_time_str,
            "end_time": end_time.strftime("%Y-%m-%d_%H%M%S"),
            "job_id": job_id,
        }
        stats_fn = f"dedup_stats.{start_time_str}.{job_id}.json"
        stats_json = json.dumps(stats, indent=4)
        logger.info(f"Stats: {stats_json}")

        with open(output_dir_path / stats_fn, "w") as f:
            f.write(stats_json)

    else:
        logger.error(f"Cannot copy dedup because hash CSV does not exist: {dedup_hash_temp_path}")

    logger.info("done")
