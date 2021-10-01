import numpy as np
import time
from numba import jit
import pandas as pd

PATH_FILE = "winequality-red.csv"


def read_data(path_file):
    DS = np.genfromtxt(path_file,delimiter=",", dtype=str)
    DS = DS[1:,:]
    return DS


def normalization(DS):
    DS = DS[:,:-1].astype(float)
    for i in range(DS.shape[1]):
        if max(DS[:,i]) > 1:
            for j in range(len(DS[:,i])):
                DS[j,i] = round((DS[j, i] - min(DS[:,i])) / (max(DS[:,i])-min(DS[:,i])),5)
    return DS


def membership_function(xi, xj, min_c, max_c):
    condition = round(abs(xi-xj)/ abs(max_c-min_c),5)
    

    if condition <= 0.25:
        result = 1 - round(4*condition,5)
    else:
        result = 0
    
    return round(result,5)

@jit(nopython=True)
def relation_matrix(DS):

    #DS_attribute = DS[:,:-1].astype(float)
    DS_attribute = normalization(DS)
    min_c = np.min(DS_attribute, axis=0)
    max_c = np.max(DS_attribute, axis=0)
    result_matrix = []
    
    
    for j in range(0,DS_attribute.shape[1]):
        matrix_attribute = np.zeros((DS_attribute.shape[0],DS_attribute.shape[0]))
        for i in range(0,DS_attribute.shape[0]):
            for h in range(0,DS_attribute.shape[0]):
                matrix_attribute[i][h] = membership_function(DS_attribute[i][j], DS_attribute[h][j], min_c[j], max_c[j])
        result_matrix.append(matrix_attribute) 
    return result_matrix


def union_matrix(r_m1, r_m2):
    union_matrix = np.zeros((r_m1.shape[0], r_m1.shape[1]))
    for i in range(0, union_matrix.shape[0]):
        for j in range(0, union_matrix.shape[1]):
            union_matrix[i][j] = min(r_m1[i][j], r_m2[i][j])
    return union_matrix


def C_matrix(result_matrix):
    C_matrix = result_matrix[0]
    for i in range(0, result_matrix.shape[0]-1):
        C_matrix = np.array(union_matrix(C_matrix, result_matrix[i+1]))
    return C_matrix


def partition(DS):
    partition_d = DS[:,-1].astype(str)
    unique_value = np.unique(partition_d)
    partition = []
    for v in unique_value:
        class_equivalent = np.where(partition_d==v)[0]
        partition.append(class_equivalent)
    return partition


def member_fuzzy(partition,r_c):
    r_c = np.array(r_c)
    result_member_fuzzy = np.zeros((r_c.shape[0], len(partition)))

    tg = lambda a,b : [b[i] if b[i] > 1-a[i] else 1-a[i] for i in range(len(a))]

    for i in range(0, r_c.shape[0]):

        for h in range(0, len(partition)):
            Muy_X_y = np.array(r_c[i].copy())
            Muy_X_y = [1 if i in partition[h] else 0 for i in range(len(Muy_X_y))]

            tg_max = (tg(r_c[i], Muy_X_y))
            result_member_fuzzy[i][h] = min(r_c[i][i], np.min(tg_max))
    return result_member_fuzzy


def signifi_pos_fuzzy(result_member_fuzzy):
    len_r_c = len(result_member_fuzzy)
    len_signifi = round(sum([max(v) for v in result_member_fuzzy]) / len_r_c,5)

    return len_signifi


def display_result(K):
    # K is a set reduct attribute
    K = list(K)
    result = set(["c{}".format(i+1) for i in K])
    return result

def process(path_file):

    
    DS = read_data(path_file)
    all_c =  [i for i in range((DS.shape[1]-1))]

    B = []
    sig = 0

    partition_d = partition(DS)
    
    ## Step 1:

    result_matrix = np.array(relation_matrix(DS))

    C = C_matrix(result_matrix)
    sig_C = signifi_pos_fuzzy(member_fuzzy(partition_d, C))

    sig_max_c = np.argmax([signifi_pos_fuzzy(member_fuzzy(partition_d,c)) for c in result_matrix])
    B.append(sig_max_c)

    print(f"step 1: Reduct Attribute is: {display_result(B)}")

    # Step2:
    step = 2
    matrix_tg = result_matrix[sig_max_c].copy()
    while (sig < sig_C):
        
        rss = []
        for i in [c for c in all_c if c not in B]:

            tg = np.array(union_matrix(matrix_tg, result_matrix[i])) 
            sig_tg = signifi_pos_fuzzy(member_fuzzy(partition_d, tg))
            
            rss.append([i,sig_tg,tg])

        rss = np.array(rss,dtype=object)
        i_kn = [rss[i] for i in range(len(rss)) if (np.argmax(rss[:,1])==i)]

        matrix_tg = i_kn[0][2]
        sig = i_kn[0][1]
        B.append(i_kn[0][0])
        print(f"step {step}: Reduct Attribute is: {display_result(B)}")
        step += 1
        
    B = set(B)
    B = display_result(B)
    
    return B


if __name__=="__main__":
    
    start_time = time.time()

    B = process(PATH_FILE)
    end_time = round(time.time() - start_time,5)
    print(f"result: {B} with: {end_time}")
