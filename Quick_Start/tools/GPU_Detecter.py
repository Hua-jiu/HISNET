import os
from time import sleep


def GPU_Detect():
    while True:
        gpu_util = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").readlines()
        gpu_util = [int(util[:-1].split()[0]) for util in gpu_util]
        id = 0
        for util in gpu_util:
            if util < 80:
                return id
            elif id == len(gpu_util)-1:
                if util > 80:
                    break
                else:
                    return id
            id += 1
        sleep(20)

