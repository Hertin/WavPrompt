#!/bin/bash

source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate wavprompt # change it to your conda environment

stage=0 # start from 0 if you need to start from data preparation
stop_stage=100
manifest_path= # path to save manifest
config_name=
subset=
save_dir=
n_token=
reduction_factor=
freeze_finetune_updates=0
all_scenarios=
output_folder=
ckpt_path_template=
split=
prompt=
manifest_dir=
ckpt_path_variable=
transcript_folder=
. utils/parse_options.sh || exit 1; # kaldi script to parse command line options

set -e
set -u
set -o pipefail

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: Training Fairseq Model (${manifest_path}) (${config_name})"
    echo "Params: freeze_finetune_updates: ${freeze_finetune_updates} reduction_factor: ${reduction_factor} n_token: ${n_token}"
    config_dir="$(pwd)/config/pretraining"
    w2v_path="$(pwd)/models/wav2vec_small.pt"
    gpt2_path="$(pwd)/models/gpt2"
    mkdir -p ${save_dir}

    if [[ -f "${save_dir}/hydra_train.log" ]]; then
        n_prev=$(ls -1q ${save_dir} | grep hydra_train | wc -l)
        echo ${n_prev} previous hydra_train.log
        mv ${save_dir}/hydra_train.log ${save_dir}/hydra_train_${n_prev}.log
        mv ${save_dir}/train.log ${save_dir}/train_${n_prev}.log
    fi

    python ${FAIRSEQ_ROOT}/fairseq_cli/hydra_train.py \
        task.data=${manifest_path} task.gpt_path=${gpt2_path} \
        model.w2v_path=${w2v_path} model.n_token=${n_token} model.reduction_factor=${reduction_factor} \
        model.gpt_path=${gpt2_path} model.freeze_finetune_updates=${freeze_finetune_updates} \
        checkpoint.save_dir=${save_dir} hydra.run.dir=${save_dir} \
        --config-dir ${config_dir} --config-name ${config_name} \
    2>&1 | tee ${save_dir}/train.log
fi

if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ]; then
    echo "Stage 100: Generating transcript for WAV2Prompt on ${manifest_dir}"
    echo """
    python -u wavprompt_generate.py \
        --manifest-dir "${manifest_dir}" --split "${split}" \
        --all-scenarios ${all_scenarios} \
        --ckpt-path-template ${ckpt_path_template} \
        --ckpt-path-variable ${ckpt_path_variable} --max-batch 1000 --transcript-folder ${transcript_folder}
    """
    python -u scripts/wavprompt_generate.py \
        --manifest-dir "${manifest_dir}" --split "${split}" \
        --all-scenarios ${all_scenarios} \
        --ckpt-path-template ${ckpt_path_template} \
        --ckpt-path-variable ${ckpt_path_variable} --max-batch 1000 --transcript-folder ${transcript_folder}

fi

if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ]; then
    echo "Stage 110: Evaluating Fairseq Model of WAV2GPT2 on ${manifest_dir}"
    echo """
    python -u scripts/wavprompt_eval.py \
        --manifest-dir "${manifest_dir}" --split "${split}" \
        --all-scenarios ${all_scenarios} \
        --prompt "${prompt}" \
        --output-folder ${output_folder} \
        --exp "base" "txt" --suffix "rf" \
        --ckpt-path-template ${ckpt_path_template} \
        --ckpt-path-variable ${ckpt_path_variable} \
        --max-batch 250"""
    python -u scripts/wavprompt_eval.py \
        --manifest-dir "${manifest_dir}" --split "${split}" \
        --all-scenarios ${all_scenarios} \
        --prompt "${prompt}" \
        --output-folder ${output_folder} \
        --exp "base" "txt" --suffix "rf" \
        --ckpt-path-template ${ckpt_path_template} \
        --ckpt-path-variable ${ckpt_path_variable} \
        --max-batch 250
fi

