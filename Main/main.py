import subprocess
from tqdm import tqdm
import time
import os, sys
import re
import pickle

project = sys.argv[1]
pp = sys.argv[1]

timing_file = f'{pp}_timing_data.txt'
if os.path.exists(timing_file):
    os.remove(timing_file)

card = [0]
try:
    lst = list(range(len(pickle.load(open(f'{project}.pkl', 'rb')))))
except FileNotFoundError:
    print(f"Error: File '{project}.pkl' not found.")
    sys.exit(1)

singlenums = {'Time':5, 'Math':2, "Lang":1, "Chart":3, "Mockito":4,
              "Closure":1, "Codec":1, 'Compress':1, 'Gson':1, 'Cli':1,
              'Jsoup':1, 'Csv':1, 'JacksonCore':1, 'JacksonXml':1,
              'Collections':1}
if project not in singlenums:
    print(f"Error: Project '{project}' not found in singlenums.")
    sys.exit(1)

singlenum = singlenums[project]
totalnum = len(card) * singlenum

lr_list = [1e-2, 1e-3, 1e-4]
batch_size_list = [32, 64, 100, 128]

lr = 1e-2
seed = 0
batch_size = 100

python_exec = sys.executable

for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn = int(j / singlenum)
        command = f"{python_exec} run.py {lst[totalnum * i + j]} {project} {lr} {seed} {batch_size}"
        try:
            p = subprocess.Popen(command, shell=True)
            jobs.append(p)
            time.sleep(10)
        except Exception as e:
            print(f"Failed to start process: {e}")
            continue

    for p in jobs:
        p.wait()

def execute_command(command):
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
        for line in p.stdout:
            print(line, end='')
        for line in p.stderr:
            print(line, end='')

execute_command(f"{python_exec} sum.py {project} {seed} {lr} {batch_size}")
execute_command(f"{python_exec} watch.py {project} {seed} {lr} {batch_size}")

training_times = []
testing_times = []
if os.path.exists(timing_file):
    with open(timing_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r"TIMING_INFO: Training Time: (\d+.\d+), Testing Time: (\d+.\d+)", line)
            if match:
                training_times.append(float(match.group(1)))
                testing_times.append(float(match.group(2)))

total_training_time = sum(training_times)
total_testing_time = sum(testing_times)
print(f"The overall training time is {total_training_time} seconds.")
print(f"The overall testing time is {total_testing_time} seconds.")
