import sys

from utils.common import arg_parser


def run(logdir, boardPath, imagePath, embeddingPath, pairs, epoch=10):
    print("KIRA is trying to use PY version: " + str(sys.version_info[0]))
    arg_parser([], logdir, boardPath, imagePath, embeddingPath, pairs, epoch, "DenseNet")


if __name__ == "__main__":
    arg_parser(sys.argv[1:])
