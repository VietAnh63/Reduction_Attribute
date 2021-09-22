
import time, math
import numpy as np
import multiprocessing as mp
from functools import partial


# # response result
# def get_result(result):
#     global results
#     results.append(result)

# def my_function(i, pr1, pr2, pr3):
#     result = pr1 ** 2 + pr2 * pr3
#     time.sleep(1)
#     return (i, result)

# def serial(i, pr1,pr2,pr3):
#     t0 = time.time()

#     for i in range(0, params.shape[0]):
#         get_result(my_function(i, params[i, 0], params[i, 1], params[i,2]))
#     print("Time in serial:", time.time()-ts)

#     return results, t1

# def parallel(N):
    
#     t0 = time.time()
#     pool = mp.Pool(mp.cpu_count())

#     #for i in range(N):
#     #   pool.apply_async(random_square, args=(i,), callback=call_back)
#     #pool.close()
#     #pool.join()
#     result = pool.map(random_square, range(N))


#     t1 = time.time() - t0
#     return result, t1
        

    

# if __name__ == "__main__":
    
#     N = 1000000

#     results1, t1 = serial(N)
#     results2, t2 = parallel(N)
#     print(t1)
#     print(t2)



# function calculate
def my_function(i, pr1, pr2, pr3):
    #time.sleep(2)
    result = pr1 ** 2 + pr2 * pr3

    return (i, result)


# response result
def get_result(result):
    global results
    results.append(result)


step = 0
# parallel
def parallel(k, N):
    global results
    #time.sleep(2)
    for i in range(len(N)):
        result = N[0] ** 2 + N[1] + N[2]

    return (k ,result)



if __name__ == "__main__":
    params = np.random.random((100,3))*100
    

    # Run serial
    ts = time.time()
    results = []
    for i in range(0, params.shape[0]):
        get_result(my_function(i, params[i, 0], params[i, 1], params[i,2]))
    print("Time in serial:", time.time()-ts)
    #print("Result", results)


    # Run Parallel
    ts_pr = time.time()

    #param = [zip(i, params[i,0], params[i,1], params[i,2]) for i in range(params.shape[0])]

    # Use number CPU
    num_cpu = mp.cpu_count()
    
    pool = mp.Pool(num_cpu)




    #rs = pool.starmap(my_function, [(j, params[j][0], params[j][1], params[j][2]) for j in range(len(params))])
    #rs = pool.map(parallel, params)
    for i in range(params.shape[0]):
        pool.apply_async(parallel, args=(i, params[i]), callback=get_result)

    pool.close()
    pool.join()


    print("Time in parallel:", time.time() - ts_pr)
    #print("Result:", rs)

