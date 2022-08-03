import os 
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year",help="Year",type=int)
parser.add_argument("-g","--gpu-type",help="GPU type")
parser.add_argument("-n","--num-gpus",help="Number of GPUs",type=int)
parser.add_argument("-m","--model",help="Model name",default="unet_19_07_2022_115722_new.pth")
parser.add_argument("-start","--start",help="Shape start idx",default="")
parser.add_argument("-end","--end",help="Shape end idx",default="")
parser.add_argument("-r","--rescale-factor",help="Rescale factor",default=1,type=int)
parser.add_argument("-cpu","--cpus",help="Number of CPUs",default=25,type=int)
parser.add_argument("-mem","--mem",help="Memory resources in GB",default=130,type=int)
parser.add_argument("-w","--num-worker",help="Number of workers for dataloading per GPU",default=20,type=int)

args = parser.parse_args()

path = "/home/c/c_luel01/satellite_data/SA_semantic_segmentation/jobs"
template = os.path.join(path,"job_template.cmd")


year = args.year
gpu_type =  args.gpu_type
n_gpus = args.num_gpus
model_name = args.model
ncpu = args.cpus
mem = args.mem
rescale = args.rescale_factor
start_idx = args.start
end_idx = args.end
num_workers = args.num_worker

if year is None or gpu_type is None or n_gpus is None:
    sys.exit("ERROR: Missing arguments! Check with --help")

tmp_file = os.path.join(path,"tmp_job_"+str(int(time.time())))+"_"+str(year)+".cmd"

with open(template) as f:
    text = f.read()

text = text.replace("$YEAR",str(year))
text = text.replace("$GPU",gpu_type)
text = text.replace("$N_GPU",str(n_gpus))
if model_name == "":
    text = text.replace("_$MODEL",model_name)
text = text.replace("$MODEL",model_name)
text = text.replace("$CPU",str(ncpu))
text = text.replace("$MEM",str(mem))
text = text.replace("$RESCALE",str(rescale))
text = text.replace("$NWORKER",str(num_workers))
text = text.replace("$START",str(start_idx))
text = text.replace("$END",str(end_idx))

with open(tmp_file, "w") as f:
    f.write(text)

os.system("sbatch "+tmp_file)
os.remove(tmp_file)
