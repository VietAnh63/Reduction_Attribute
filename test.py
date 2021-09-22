import numpy as np
import time
import multiprocessing as mp


def random_square(seed):
    np.random.seed(seed)
    random_num = np.random.randint(0, 10)
    return random_num**2

# t0 = time.time()
# results = []
# for i in range(10000000): 
#     results.append(random_square(i))
# t1 = time.time()
# print(f'Execution time {t1 - t0} s')

t0 = time.time()
n_cpu = mp.cpu_count()

pool = mp.Pool(processes=n_cpu)
results = [pool.map(random_square, range(10000000))]
t1 = time.time()
print(f'Execution time {t1 - t0} s')