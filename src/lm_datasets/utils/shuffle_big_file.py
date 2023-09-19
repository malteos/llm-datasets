"""
Take from https://raw.githubusercontent.com/YaYaB/shuffle-big-file/master/shuffle_big_file/shuffle_big_file.py

This tool helps you shuffle by line a big file that does not fit in memory. Given the
batch_size that your machine can put in memory it will shuffle the whole file by reading
as many times as necessary.
"""
import os
from datetime import datetime, timedelta
import math
import random
import argparse
import sys
import logging

RANDOM_SEED = 1337

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser("Shuffle a huge file without loading in fully in RAM")
    parser.add_argument("--input_file", type=str, help="Path to input file")
    parser.add_argument("--batch_size", type=int, help="Batch size that can fit in memory")
    parser.add_argument("--output_file", type=str, help="Path to output shuffled file")
    parser.add_argument("--seed", type=int, help="Seed for random generator", default=RANDOM_SEED)

    args = parser.parse_args()
    return args


def compute_nb_read(path_file, batch_size):
    """
    :param path_file: path file for which we count the number of lines
    :param batch_size: batch size that the machine can handle
    :return: number of lines and number of read necessary
    """
    # Read all file for the first time to know how many read we need
    with open(path_file) as f:
        for i, l in enumerate(f):
            pass

    nb_lines = i + 1
    nb_read = math.ceil(nb_lines / batch_size)

    return nb_read, nb_lines


def compute_shuffled_index(nb_lines):
    """
    :param nb_lines: number of lines
    :return: shuffled index of the number of lines
    """
    # Create index
    idx_shuffled = list(range(0, nb_lines))

    # Shuffle the index
    random.shuffle(idx_shuffled)

    return idx_shuffled


def generate_random_string(
    length=500, alphabet="azertyuiop^$qsdfghjklmù*wxcvbn,;:!1234567890°+¨£%µ§/.?<>AZERTYUIOPQSDFGHJKLMWXCVBN&é'(-è_çà)="
):
    """
    :param length: length of string wanted
    :param alphabet: alphabet from witch characters are sampled
    :return: random string of size `length`
    """
    return "".join(random.choice(alphabet) for x in range(length))


def generate_random_file(path_file="./random_file.txt", nb_lines=10000, max_line_length=1000):
    """
    :param path_file: path file that will be created
    :param nb_lines: number of lines wanted
    :param max_lines_length: maximum number of characters per line
    """
    with open(path_file, "w") as f:
        for i in range(0, nb_lines):
            f.write(generate_random_string(max_line_length) + "\n")


def generate_random_file_cli():
    parser = argparse.ArgumentParser("generate random file containing a string per line")
    parser.add_argument("--path_file", type=str, default="./random_file.txt", help="Path to file that will be created")
    parser.add_argument("--nb_lines", type=int, default=10000, help="Number of lines wanted for the file")
    parser.add_argument("--max_line_length", type=int, default=1000, help="Max number of characters per line")
    opt = parser.parse_args()

    generate_random_file(opt.path_file, opt.nb_lines, opt.max_line_length)


def shuffle(input_file, output_file, idx_shuffled, nb_read, batch_size):
    """
    :param input_file: path file that will be created
    :param output_file: number of lines wanted
    :param idx_shuffled: index shuffled
    :param nb_read: number of read necessary
    :param batch_size: batch size used

    """
    with open(output_file, "a") as f:
        for i in range(nb_read):
            startTime = datetime.now()
            elements = []
            idx = idx_shuffled[i * batch_size : (i + 1) * batch_size]
            # get index of batch in order
            idx_sort = sorted(idx)
            waiting_line = idx_sort[0]

            j = 0
            with open(input_file) as in_f:
                for k, l in enumerate(in_f):
                    # If line in batch add it to list of elements
                    if k == waiting_line:
                        elements.append(l)
                        j += 1
                        # Try to access element to the idx sort
                        try:
                            waiting_line = idx_sort[j]
                        # If fails that means it exceds the max_line
                        except:
                            break

            # Shuffle the batch created
            random.shuffle(elements)

            # Append output file
            for elem in elements:
                f.write(elem)

            logger.info(
                f"{nb_read - (i+1)} read are left. It will take"
                f" {timedelta(seconds=(nb_read - (i+1)) * (datetime.now() - startTime).total_seconds())}"
            )


def shuffle_big_file(input_file, output_file, batch_size, seed=RANDOM_SEED):
    """
    :param input_file: path of the inputfile
    :param batch_size: batch size used
    :param output_file: path of the output file
    :param seed: seed for the random generator
    """

    assert os.path.exists(input_file)
    assert not os.path.exists(output_file)

    random.seed(seed)

    # Compute number of read necessary
    startTime_ = datetime.now()
    nb_read, nb_lines = compute_nb_read(input_file, batch_size)
    one_read = datetime.now() - startTime_

    logger.info(
        f"File was read in {one_read} and {nb_read} reads are necessary. It will take"
        f" {timedelta(seconds=nb_read * one_read.total_seconds())}"
    )

    # Shuffle the index of the number of lines
    idx_shuffled = compute_shuffled_index(nb_lines)

    # Shuffle file
    shuffle(input_file, output_file, idx_shuffled, nb_read, batch_size)

    logger.info(f"File was shuffled in {datetime.now() - startTime_}")

    return 0


def shuffle_big_file_cli():
    """
    Main function of the program
    """

    # Get arguments passed in CLI
    opt = get_args()

    return shuffle_big_file(opt.input_file, opt.output_file, opt.batch_size, opt.seed)


if __name__ == "__main__":
    try:
        sys.exit(shuffle_big_file_cli())
    except Exception as e:
        print("[ERR] Uncaught error waiting for scripts to finish")
        print(e)
        raise
