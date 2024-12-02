#! /bin/bash
#PBS -N Rec_Training
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=1
#PBS -q gpu
#PBS -P Movie-Recommendation-Model

module load compiler/anaconda3
source /home/kartikeya.agrawal_ug25/Movie-Recommendation/movie-rec/bin/activate
python3 /home/kartikeya.agrawal_ug25/Movie-Recommendation/k_means_labelling.py
python3 /home/kartikeya.agrawal_ug25/Movie-Recommendation/get_recommendation.py
python3 /home/kartikeya.agrawal_ug25/Movie-Recommendation/convert_recommendations.py
