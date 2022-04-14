#!/bin/bash

stage=0
stop_stage=3

PYTHON_ENVIRONMENT=wavprompt
CONDA_ROOT=/nobackup/users/$(whoami)/espnet/tools/conda # change it to your conda root

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh

cwd=$(pwd)
FAIRSEQ=${cwd}/fairseq/fairseq
CODE=${cwd}/wavprompt/fairseq

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Install conda environment..."
    # remove the environment named wavprompt
    # conda remove --name ${PYTHON_ENVIRONMENT} --all -y
    
    # install environment and name it wavprompt
    conda create --name ${PYTHON_ENVIRONMENT} python=3.7 -y
fi

conda activate ${PYTHON_ENVIRONMENT}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Install fairseq dependencies..."
    module load cuda/10.2
    conda install pytorch=1.7.1 -y --name ${PYTHON_ENVIRONMENT}
    
    if [ ! -d fairseq ]; then
        # clone fairseq v0.10.2
        git clone https://github.com/pytorch/fairseq.git --branch main --single-branch
    fi
    cd ${cwd}/fairseq
    # checkout the fairseq version to use
    git reset --hard 1bba712622b8ae4efb3eb793a8a40da386fe11d0
    
    # optionally disable torchaudio as we are not using it
    mv setup.py setup.py.bak
    sed '/"torchaudio>=0.8.0"/s/^/#/' setup.py.bak > setup.py

    if [ $(pip freeze | grep fairseq | wc -l ) -gt 0 ]; then
        echo "Already installed fairseq. Skip..."
    else
        echo "Install fairseq..."
        python -m pip install --editable ./
    fi
    # optionally do this if related error occurs
    python setup.py build_ext --inplace
    
    python -m pip install soundfile
    python -m pip install transformers==4.9.1
    python -m pip install tensorboardX
    python -m pip install editdistance
    python -m pip install easydict
    python -m pip install pandas
    cd ${cwd}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Copy model files to fairseq..."
    mkdir -p ${cwd}/wavprompt/models/gpt2
    # download wav2vec base model
    cp ${CODE}/criterions/cross_entropy_with_accuracy.py ${FAIRSEQ}/criterions
    
    cp ${CODE}/data/add_target_dataset_wavprompt.py ${FAIRSEQ}/data
    cp ${CODE}/data/add_target_dataset_wavprompt_evaluation.py ${FAIRSEQ}/data
    cp ${CODE}/data/audio/file_audio_label_dataset.py ${FAIRSEQ}/data/audio
    if [ -f ${FAIRSEQ}/data/__init__.py ] && [ $(cat ${FAIRSEQ}/data/__init__.py | grep """$(cat ${CODE}/data/__init__.py)""" | wc -l) != $(cat ${CODE}/data/__init__.py | wc -l) ]; then
        (echo ""; cat ${CODE}/data/__init__.py) >> ${FAIRSEQ}/data/__init__.py
    fi

    cp -r ${CODE}/models/wavprompt ${FAIRSEQ}/models
    
    cp ${CODE}/tasks/wavprompt_pretraining.py ${FAIRSEQ}/tasks
    cp ${CODE}/tasks/wavprompt_evaluation.py ${FAIRSEQ}/tasks
    
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Download to pretrained wav2vec..."
    mkdir -p ${cwd}/wavprompt/models/gpt2
    # download wav2vec base model
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O ${cwd}/wavprompt/models/wav2vec_small.pt
    
fi
