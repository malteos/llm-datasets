import io
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable
from pyarrow import parquet as pq
from lm_datasets.utils import get_auto_workers
from lm_datasets.utils.config import Config
import sentencepiece as spm
from transformers.convert_slow_tokenizer import import_protobuf


def train_sp_tokenizer(config: Config):
    logger = config.init_logger(__name__)

    # from_pretrained_model_path = ""  # existing SP model
    tokenizer_train_ratio = config.tokenizer_ratio
    text_field_name = (
        config.text_field_name
    )  # "dataset_id"  # bugfix <--- dataset was created with bad column names "dataset_id" and "text" are mixed up
    batch_size = config.input_batch_size
    composed_dataset_path = Path(config.composed_dataset_dir)
    train_stats_path = composed_dataset_path / "train_stats.json"
    output_tokenizer_path = Path(config.output_tokenizer_path)
    train_pq_file_paths = list(sorted(composed_dataset_path.glob("train.part-*.parquet")))

    if not composed_dataset_path.exists():
        raise FileNotFoundError("Composed dataset path does not exists: %s" % composed_dataset_path)

    if output_tokenizer_path.exists() and not config.override:
        raise FileExistsError("Output exists already at: %s (fix with --overide)" % output_tokenizer_path)

    if not output_tokenizer_path.parent.exists():
        os.makedirs(output_tokenizer_path.parent)

    if not train_stats_path.exists():
        raise FileNotFoundError("Training data stats file does not exists: %s" % train_stats_path)

    with open(train_stats_path) as f:
        train_stats = json.load(f)

    dataset_id_to_train_row_count = {
        ds_id: stats["split_to_offset_and_limit"]["train"][1] - stats["split_to_offset_and_limit"]["train"][0]
        for ds_id, stats in train_stats["dataset_id_to_stats"].items()
    }
    total_train_row_count = sum(dataset_id_to_train_row_count.values())
    tokenizer_train_row_count = math.floor(tokenizer_train_ratio * total_train_row_count)

    logger.info("Dataset has %i total rows", total_train_row_count)

    def generate_texts(pq_file_paths, row_limit=0, text_column="text", sentence_splitting: bool = False) -> Iterable:
        """
        Stream text data from parquet files
        """
        pq_dataset = pq.ParquetDataset(path_or_paths=pq_file_paths)

        logger.info("Dataset initialized with %i fragments", len(pq_dataset.fragments))

        rows = 0
        for frament in pq_dataset.fragments:
            for batch in frament.to_batches(batch_size=batch_size, columns=[text_column]):
                for text in batch.columns[0]:
                    text = text.as_py()  # cast to py

                    if sentence_splitting:
                        for sentence in text.splitlines():  # very basic sentence splitting. TODO is this OK for code?
                            yield sentence

                            # rows += 1

                            # if row_limit > 0 and rows >= row_limit:
                            #     # sentence loop
                            #     break
                    else:
                        yield text

                    rows += 1

                    if row_limit > 0 and rows >= row_limit:
                        # text loop
                        break

                if row_limit > 0 and rows >= row_limit:
                    # batch loop
                    break

            if row_limit > 0 and rows >= row_limit:
                # frament loop
                break

        logger.info("Texts generated: %i", rows)

    # Source tokenizer
    if config.source_tokenizer_path is not None:
        model_pb2 = import_protobuf()

        source_model_proto = model_pb2.ModelProto()
        with open(config.source_tokenizer_path, "rb") as f:
            source_model_proto.ParseFromString(f.read())
        # source_proto = source_model_proto

        # print(source_model_proto.trainer_spec)
        source_trainer_spec = source_model_proto.trainer_spec

        source_trainer_spec_keys = [f.name for f in source_trainer_spec.DESCRIPTOR.fields]
        source_trainer_spec_dict = {k: getattr(source_trainer_spec, k) for k in source_trainer_spec_keys}

        logger.info("Source tokenizer loaded from: %s", config.source_tokenizer_path)
        logger.info("Source train specs: %s", source_trainer_spec_dict)

    else:
        logger.info("Source tokenizer was not provided")

        source_trainer_spec_dict = {}

    # Text generator
    logger.info("Tokenizer will be trained on %i input rows", tokenizer_train_row_count)

    texts = generate_texts(
        train_pq_file_paths,
        row_limit=tokenizer_train_row_count,
        text_column=text_field_name,
        sentence_splitting=config.sentence_splitting,
    )

    # Write model to memory
    model_writer = io.BytesIO()

    # Trainer settings

    # These settings are loaded from the source model
    allowed_source_kwargs = [
        # user settings
        # "vocab_size",
        # "num_threads",
        # "model_type",
        # options
        # "enable_differential_privacy",
        "self_test_sample_size",
        # "differential_privacy_noise_level",
        # "differential_privacy_clipping_threshold",
        "character_coverage",
        "input_sentence_size",
        "shuffle_input_sentence",
        "seed_sentencepiece_size",
        "shrinking_factor",
        "max_sentence_length",
        "num_sub_iterations",
        "max_sentencepiece_length",
        "split_by_unicode_script",
        "split_by_number",
        "split_by_whitespace",
        "treat_whitespace_as_suffix",
        # "allow_whitespace_only_pieces",
        "split_digits",
        "byte_fallback",
        "vocabulary_output_piece_score",
        "hard_vocab_limit",
        "use_all_vocab",
        "unk_id",
        "bos_id",
        "eos_id",
        "pad_id",
        "unk_piece",
        "bos_piece",
        "eos_piece",
        "pad_piece",
        "unk_surface",
        "train_extremely_large_corpus",
    ]

    # Default settings (based on Mistral)
    trainer_kwargs = dict(
        # User settings (via config)
        sentence_iterator=texts,
        model_writer=model_writer,
        num_threads=get_auto_workers(
            config.workers
        ),  # BPE doesn't support multi-threaded training. num_threads are not used.
        model_type=config.tokenizer_model_type,  # model algorithm: unigram, bpe, word or char
        vocab_size=config.tokenizer_vocab_size,
        # Mistral settings
        enable_differential_privacy=False,
        self_test_sample_size=0,
        differential_privacy_noise_level=0.0,
        differential_privacy_clipping_threshold=0,
        character_coverage=0.9999499917030334,
        input_sentence_size=200_000_000,  # (maximum size of sentences the trainer loads), default: 0
        shuffle_input_sentence=True,  # (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)
        # mining_sentence_size=0,  # unkown
        # training_sentence_size=0,
        seed_sentencepiece_size=1000000,  # sp default
        shrinking_factor=0.75,  # sp default
        max_sentence_length=4192,  # sp default
        # num_threads=80,
        num_sub_iterations=2,  # sp default
        max_sentencepiece_length=16,  # sp default
        split_by_unicode_script=True,  # sp default
        split_by_number=True,  # sp default
        split_by_whitespace=True,  # sp default
        treat_whitespace_as_suffix=False,  # sp default
        allow_whitespace_only_pieces=True,
        split_digits=True,
        # pretokenization_delimiter='',  # unknown
        # 'control_symbols': [], 'user_defined_symbols': [], 'required_chars': '',
        byte_fallback=True,
        vocabulary_output_piece_score=True,
        hard_vocab_limit=True,  # sp default
        use_all_vocab=False,  # sp default
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        unk_surface=" ⁇ ",  # sp default
        train_extremely_large_corpus=False,  # sp default
        # model_prefix=model_prefix,  # `model_prefix`and `model_writer` cannot be defined at the same time!
    )

    # set settings from source tokenizer
    if source_trainer_spec_dict:
        for k in allowed_source_kwargs:
            trainer_kwargs[k] = source_trainer_spec_dict[k]

    start_time = time.perf_counter()

    logger.info("SP train settings: %s", trainer_kwargs)

    # train with combination of user settings and source model settings
    train_res = spm.SentencePieceTrainer.train(
        **trainer_kwargs,
    )

    # Serialize the model as file
    with open(output_tokenizer_path, "wb") as f:
        f.write(model_writer.getvalue())

    logger.info("Saved to %s", config.output_tokenizer_path)

    # Directly load the model from serialized model.
    sp = spm.SentencePieceProcessor(model_proto=model_writer.getvalue())
    logger.info("Example output: %s", sp.encode("this is test"))

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"done (training took {elapsed_time=:0.2f} seconds)")
