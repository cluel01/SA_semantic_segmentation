import os 
import sys
import time

path = "/home/c/c_luel01/satellite_data/SA_semantic_segmentation/jobs"
template = os.path.join(path,"job_template.cmd")

if len(sys.argv) != 5:
    sys.exit("Missing arguments: start_jobs.py <AREA> <YEAR> <GPU_PARTITION> <N_GPUS>")

area = str(sys.argv[1])
year = str(sys.argv[2])
gpu_type =  str(sys.argv[3])
n_gpus = str(sys.argv[4])

if int(area) not in range(22,35):
    sys.exit("Not existing area!")

tmp_file = os.path.join(path,"tmp_job_"+str(int(time.time())))+"_"+str(year)+".cmd"

with open(template) as f:
    text = f.read()

text = text.replace("$AREA",area)
text = text.replace("$YEAR",year)
text = text.replace("$GPU",gpu_type)
text = text.replace("$N_GPU",n_gpus)


with open(tmp_file, "w") as f:
    f.write(text)

os.system("sbatch "+tmp_file)
os.remove(tmp_file)