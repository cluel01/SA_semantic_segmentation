#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=$CPU
#SBATCH --partition=$GPU
#SBATCH --gres=gpu:$N_GPU
#SBATCH --time=48:00:00
#SBATCH --mem=$MEMG

#SBATCH --job-name=inference_$YEAR_$GPU_$MODEL
#SBATCH --output=logs/output_$YEAR_$MODEL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christian.luelf@uni-muenster.de
 
# load modules with available GPU support (this is an example, modify to your needs!)
ml load palma/2020b
ml load fosscuda
ml load GDAL/3.2.1
ml load PyTorch/1.9.0
#ml load OpenCV

# run the application
python /home/c/c_luel01/satellite_data/SA_semantic_segmentation/jobs/run_inference.py $YEAR $N_GPU $GPU $MODEL $RESCALE $NWORKER $START $END
