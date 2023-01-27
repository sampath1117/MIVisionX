import os
import numpy as np
numfiles = 128
path = "/media/sampath/tensor_debug/sampath_MIVisionX/rocAL/rocAL_pybind/example/"
for i in range(numfiles):
    cpu_file_name = path + "cpu/matches_cpu_" + str(i) + ".txt"
    gpu_file_name = path + "gpu/matches_gpu_" + str(i) + ".txt"
    cpu_data = np.loadtxt(cpu_file_name, dtype=int)
    gpu_data = np.loadtxt(gpu_file_name, dtype=int)
    result = np.absolute(cpu_data - gpu_data)
    colors, counts = np.unique(result, return_counts = True, axis = 0)
    print("differences for file: ",i)
    for color, count in zip(colors, counts):
        if color != 0:
            print(color, count)

