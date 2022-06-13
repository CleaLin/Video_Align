echo "Motion Analysis for $1 in $2 mode"
# Source your conda.sh file
CONDA="/home/clealin/anaconda3/etc/profile.d/conda.sh"
source $CONDA

# Run AlphaPose
conda deactivate
conda activate alphapose
cd alpha_pose/
rm -rf ./outputs/*
python3 videopose.py
conda deactivate

# Copy AlpphaPose result to TCC data folder
cd ..
A_OUTPUT='./alpha_pose/outputs'
TCC_DATA="./data/$1/$2"
rm -rf $TCC_DATA/*
find $A_OUTPUT -maxdepth 1 -mindepth 1 -exec cp -r "{}" $TCC_DATA \;

# Run TCC alignment
conda activate tcc
python3 tcc_get_start.py --dataset $1 --mode $2
conda deactivate