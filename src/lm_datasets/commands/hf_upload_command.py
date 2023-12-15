from argparse import Namespace, _SubParsersAction

from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config

import os
from pathlib import Path
from huggingface_hub import HfApi
from tqdm.auto import tqdm


class HFUploadCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser("hf_upload", help="Upload files or directories to Huggingface Hub.")

        subcommand_parser.add_argument("input_path", help="Files from this directory are uploaded")
        subcommand_parser.add_argument("repo_id", help="HF hub repo ID (e.g., username/dataset_name)")
        subcommand_parser.add_argument("--path_in_repo", type=str, default=".", help="Path in HF repo")
        subcommand_parser.add_argument("--repo_type", type=str, default="dataset", help="Path in HF repo")
        subcommand_parser.add_argument(
            "--commit_message", type=str, default="Files uploaded", help="Git commit message"
        )
        subcommand_parser.add_argument(
            "--skip_if_exists", action="store_true", help="Skip if a file it already exists in the HF repo"
        )
        subcommand_parser.add_argument(
            "--input_glob", type=str, default=None, help="Glob pattern to selected matching files from input directory"
        )
        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            log=True,
        )
        subcommand_parser.set_defaults(func=HFUploadCommand)

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config = Config(**args.__dict__)

    def run(self) -> None:
        logger = self.config.init_logger(__name__)
        args = self.args

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
