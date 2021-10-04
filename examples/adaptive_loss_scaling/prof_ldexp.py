import numpy as np
import cupy as cp
from timeit import default_timer as timer


def main():
    A = cp.random.random(size=(4096, 4096)).astype(cp.float16)

    start = timer()
    for _ in range(10000):
        B1 = cp.ldexp(A, 15)
    end = timer()
    print("Elapsed: {:.6f}s".format(end - start))
    print(B1)

    start = timer()
    for _ in range(10000):
        B2 = A * np.float16(2 ** 15)
    end = timer()
    print("Elapsed: {:.6f}s".format(end - start))

    start = timer()
    for _ in range(10000):
        B3 = (A * np.float32(2 ** 15)).astype(np.float16)
    end = timer()
    print("Elapsed: {:.6f}s".format(end - start))

    print(np.sum(np.abs(B2 - B1)))


if __name__ == "__main__":
    main()
