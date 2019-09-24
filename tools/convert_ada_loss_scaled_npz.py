""" Convert the snapshot file """

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog='Convert snapshot file created from AdaLossScaled models.')
    parser.add_argument('-i', '--input-file', help='Input file', type=str)
    parser.add_argument('-o', '--output-path', help='Output file', type=str)
    args = parser.parse_args()

    # perform the update
    snapshot = np.load(args.input_file)
    data = {}

    for key in snapshot.files:
        key_ = key.replace('/link', '').replace('/residual', '/residual_conv')
        data[key_] = snapshot[key]

    np.savez(args.output_path, **data)


if __name__ == '__main__':
    main()
