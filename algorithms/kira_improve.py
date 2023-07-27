import sys
import os
import numpy as np
from utils.common import arg_parser


def run(logdir, boardPath, imagePath, embeddingPath, pairs, epoch=10):
    print("KIRA is trying to use PY version: " + str(sys.version_info[0]))
    pairs = np.load(os.path.join(logdir, 'trainParams.npy'))
    print("-------- BOARD PATH ---------- <<<<<<<<<<<<<<<<<<<")
    print(embeddingPath)
    arg_parser([], logdir, boardPath, imagePath, embeddingPath, pairs, epoch, "VGG", True)


if __name__ == "__main__":
    arg_parser(sys.argv[1:])
