import os
import numpy as np
import argparse
import copy

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('snapshot_dir', type=str)
  args = parser.parse_args()

  npz_file = np.load(os.path.join(args.snapshot_dir, 'snapshot_model.npz'))
  d = {}

  for key in npz_file.keys():
    val = npz_file[key]
    new_key = key.replace('link/', '')
    print(key, new_key)
    d[new_key] = val 

  np.savez(os.path.join(args.snapshot_dir, 'new_snapshot_model.npz'), **d)


if __name__ == '__main__':
  main()
