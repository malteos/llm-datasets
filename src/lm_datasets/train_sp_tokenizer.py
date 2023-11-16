import argparse
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable
from pyarrow import parquet as pq
from lm_datasets.utils.config import get_common_argparser, parse_args_and_get_config
import sentencepiece as spm
from transformers.convert_slow_tokenizer import import_protobuf
from tqdm.auto import tqdm

from lm_datasets.utils.settings import DEFAULT_TOKENIZER_RATIO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser(required_configs=True)], add_help=False)
    parser.add_argument(
        "--composed_dataset_dir",
        required=True,
        type=str,
        help="""Save composed dataset this directory""",
    )
    parser.add_argument(
        "--source_tokenizer_path",
        default=None,
        type=str,
        help="Source tokenizer is loaded from this path (.model file)",
    )
    parser.add_argument(
        "--output_tokenizer_path",
        default=None,
        type=str,
        help="SP tokenizer model is saved to this path",
    )
    parser.add_argument(
        "--text_field_name",
        default="text",
        type=str,
        help="Text is read from this field name from composed dataset files.",
    )
    parser.add_argument(
        "--tokenizer_ratio",
        default=DEFAULT_TOKENIZER_RATIO,
        type=float,
        help="This ratio of the training set is used for the tokenizer training.",
    )
    parser.add_argument(
        "--input_batch_size",
        default=10_000,
        type=int,
        help="Input data is read with this batch size",
    )
    parser.add_argument("--override", action="store_true", help="Override existing output files")

    config = parse_args_and_get_config(parser)

    logger = config.init_logger(__name__)

    # from_pretrained_model_path = ""  # existing SP model
    tokenizer_train_ratio = config.tokenizer_ratio
    text_field_name = config.text_field_name  # "dataset_id"  # bugfix <--- dataset was created with bad column names "dataset_id" and "text" are mixed up
    batch_size = config.input_batch_size
    composed_dataset_path = Path(config.composed_dataset_dir)
    output_tokenizer_path = Path(config.output_tokenizer_path)
    train_pq_file_paths = list(sorted(composed_dataset_path.glob("train.part-*.parquet")))

    if output_tokenizer_path.exists() and not config.override:
        raise FileExistsError("Output exists already at: %s (fix with --overide)" % output_tokenizer_path)

    if not output_tokenizer_path.parent.exists():
        os.makedirs(output_tokenizer_path.parent)

    with open(composed_dataset_path / "train_stats.json") as f:
        train_stats = json.load(f)

    dataset_id_to_train_row_count = {ds_id: stats["split_to_offset_and_limit"]["train"][1] - stats["split_to_offset_and_limit"]["train"][0] for ds_id, stats in train_stats["dataset_id_to_stats"].items()}
    total_train_row_count = sum(dataset_id_to_train_row_count.values())
    tokenizer_train_row_count = math.floor(tokenizer_train_ratio * total_train_row_count)

    logger.info("Dataset has %i total rows", total_train_row_count)

    def generate_texts(pq_file_paths, limit=0, text_column="text") -> Iterable:
        """
        Stream text data from parquet files
        """
        pq_dataset = pq.ParquetDataset(path_or_paths=pq_file_paths)

        logger.info("Dataset initialized with %i fragments", len(pq_dataset.fragments))

        rows = 0
        for frament in pq_dataset.fragments:
            for batch in frament.to_batches(batch_size=batch_size, columns=[text_column]):
                for text in batch.columns[0]:
                    yield text.as_py()  # cast to py

                    rows += 1

                    if limit > 0 and rows >= limit:
                        # row loop
                        break

                if limit > 0 and rows >= limit:
                    # batch loop
                    break


            if limit > 0 and rows >= limit:
                # frament loop
                break

        logger.info("Texts generated: %i", rows)

    # texts = list(generate_texts(train_pq_file_paths, limit=5, text_column=text_field_name))

    # Source tokenizer
    model_pb2 = import_protobuf()

    source_model_proto = model_pb2.ModelProto()
    with open(config.source_tokenizer_path, "rb") as f:
        source_model_proto.ParseFromString(f.read())
    # source_proto = source_model_proto

    # print(source_model_proto.trainer_spec)
    trainer_spec = source_model_proto.trainer_spec

    trainer_spec_keys = [f.name for f in trainer_spec.DESCRIPTOR.fields]
    trainer_spec_dict = {k: getattr(trainer_spec, k) for k in trainer_spec_keys}

    logger.info("Source tokenizer loaded from: %s", config.source_tokenizer_path)
    logger.info("Source train specs: %s", trainer_spec_dict)

    # model_prefix = "foobar"
    model_writer = io.BytesIO()

    trainer_kwargs = {}

    # source args
    # trainer_kwargs.update(trainer_spec_dict)

    # remove_fields_from_source = [
    #     "input",
    #     "input_format",
    #     "mining_sentence_size",
    #     "model_type",
    #     "training_sentence_size",
    # ]
    # for k in remove_fields_from_source:
    #     del trainer_kwargs[k]
    logger.info("Tokenizer will be trained on %i rows", tokenizer_train_row_count)
    texts = generate_texts(train_pq_file_paths, limit=tokenizer_train_row_count, text_column=text_field_name)
    # texts = tqdm(texts, total=tokenizer_train_row_count, desc="Generating text")
    # texts = list(texts)

    """
    {'input': ['/mnt/test/datasets/tokenizer_training/8T_train_data/shuffled.txt'], 'input_format': 'text', 'model_prefix': 'tok_v0', 'model_type': 2, 'vocab_size': 32000, 'accept_language': [], 'self_test_sample_size': 0, 'enable_differential_privacy': False, 'differential_privacy_noise_level': 0.0, 'differential_privacy_clipping_threshold': 0, 'character_coverage': 0.9999499917030334, 'input_sentence_size': 200000000, 'shuffle_input_sentence': True, 'mining_sentence_size': 0, 'training_sentence_size': 0, 'seed_sentencepiece_size': 1000000, 'shrinking_factor': 0.75, 'max_sentence_length': 4192, 'num_threads': 80, 'num_sub_iterations': 2, 'max_sentencepiece_length': 16, 'split_by_unicode_script': True, 'split_by_number': True, 'split_by_whitespace': True, 'treat_whitespace_as_suffix': False, 'allow_whitespace_only_pieces': True, 'split_digits': True, 'pretokenization_delimiter': '', 'control_symbols': [], 'user_defined_symbols': [], 'required_chars': '', 'byte_fallback': True, 'vocabulary_output_piece_score': True, 'hard_vocab_limit': True, 'use_all_vocab': False, 'unk_id': 0, 'bos_id': 1, 'eos_id': 2, 'pad_id': -1, 'unk_piece': '<unk>', 'bos_piece': '<s>', 'eos_piece': '</s>', 'pad_piece': '<pad>', 'unk_surface': ' ⁇ ', 'train_extremely_large_corpus': False}
    """

    # override with new args
    # see https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/doc/options.md
    trainer_kwargs.update(dict(
        sentence_iterator=texts,
        vocab_size=250680,  # divisble by 8 - Jan's recommendation
        # NVIDIA: 256000
        model_writer=model_writer,
        num_threads=70,
        # Mistral settings
        model_type="bpe",  # model algorithm: unigram, bpe, word or char
        enable_differential_privacy=False,
        self_test_sample_size=0,
        differential_privacy_noise_level=0.0,
        differential_privacy_clipping_threshold=0,
        character_coverage=0.9999499917030334,
        input_sentence_size=200000000,
        shuffle_input_sentence=True,
        # mining_sentence_size=0,  # unkown
        # training_sentence_size=0,
        seed_sentencepiece_size=1000000, # sp default
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
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        pad_piece='<pad>',
        unk_surface=' ⁇ ',  # sp default
        train_extremely_large_corpus=False,  # sp default
        # model_prefix=model_prefix,  # `model_prefix`and `model_writer` cannot be defined at the same time!
    ))

    start_time = time.perf_counter()

    # train with the same settings as in the source tokenizer
    train_res = spm.SentencePieceTrainer.train(
        **trainer_kwargs,
    )

    # Serialize the model as file
    with open(output_tokenizer_path, 'wb') as f:
      f.write(model_writer.getvalue())

    logger.info("Saved to %s", config.output_tokenizer_path)

    # Directly load the model from serialized model.
    sp = spm.SentencePieceProcessor(model_proto=model_writer.getvalue())
    logger.info("Example output: %s", sp.encode("this is test"))

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"done (training took {elapsed_time=:0.2f} seconds)")
