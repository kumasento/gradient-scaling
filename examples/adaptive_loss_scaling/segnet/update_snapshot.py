import os
import numpy as np
import argparse
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=str)
    args = parser.parse_args()

    npz_file = np.load(args.snapshot)
    d = {}

    for key in npz_file.keys():
        val = npz_file[key]
        new_key = key.replace("link/", "")
        print(key, new_key)
        d[new_key] = val

    file_name = args.snapshot.split("/")[-1]
    new_file_name = "new_" + file_name
    np.savez(os.path.join(os.path.dirname(args.snapshot), new_file_name), **d)


if __name__ == "__main__":
    main()
