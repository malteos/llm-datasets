"""
Upload files to HF hub via this script.

You need to login first:

huggingface-cli login --token $HUGGINGFACE_TOKEN

Example usage:

python hf_upload_to_hub.py ./data/dummy hello-world-org/some-data-v1 --path_in_repo=data
"""
import argparse
import logging
import os
from pathlib import Path
from huggingface_hub import HfApi
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Files from this directory are uploaded")
    parser.add_argument("repo_id", help="HF hub repo ID (e.g., username/dataset_name)")
    parser.add_argument("--path_in_repo", type=str, default=".", help="Path in HF repo")
    parser.add_argument("--repo_type", type=str, default="dataset", help="Path in HF repo")
    parser.add_argument("--commit_message", type=str, default="Files uploaded", help="Git commit message")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if a file it already exists in the HF repo")
    parser.add_argument(
        "--input_glob", type=str, default=None, help="Glob pattern to selected matching files from input directory"
    )
    args = parser.parse_args()
    api = HfApi()

    input_path = args.input_path

    # is directory
    if os.path.isdir(input_path):
        if args.input_glob is None:
            # all files from input directory
            input_file_names = list(sorted(os.listdir(input_path)))

        else:
            # select files based on glob pattern
            input_file_names = [p.name for p in sorted(Path(input_path).glob(args.input_glob))]

    elif os.path.isfile(input_path):
        # single file upload
        input_file_names = [Path(input_path).name]
        input_path = Path(input_path).parent

    repo_type = args.repo_type

    logger.info(f"Starting upload to {args.repo_id} from {input_path}")

    for i, file_name in tqdm(enumerate(input_file_names, 1), total=len(input_file_names), desc="Uploading"):
        path_in_repo = f"{args.path_in_repo}/{file_name}"

        if args.skip_if_exists:
            if api.file_exists(repo_id=args.repo_id, filename=path_in_repo, repo_type=repo_type):
                logger.info("Skip file because it already exists in the repo: %s", path_in_repo)
                continue

        api.upload_file(
            path_or_fileobj=os.path.join(input_path, file_name),
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            commit_message=args.commit_message + f" ({i}/{len(input_file_names)})",
            repo_type=repo_type,
        )
    logger.info("done")
