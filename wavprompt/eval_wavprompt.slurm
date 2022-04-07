#!/bin/bash
#SBATCH -J evalwav2gpt2
#SBATCH -o evalwav2gpt2_%j.out
#SBATCH -e evalwav2gpt2_%j.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0


## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=wavprompt
CONDA_ROOT=/nobackup/users/$(whoami)/espnet/tools/conda

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date


# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")

# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspsing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")


# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspgenderasrsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspgenderasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")

# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspgenderasrsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspgenderasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspgenderasrsing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspgenderasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspgenderasrsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspgenderasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")


# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32"

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variable="8 16 32"

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")
# ckpt_path_variable="8 16 32"


# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32"

# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32 2"
# for i in "${!all_scenarios[@]}"; do
#     all_scena=${all_scenarios[$i]}
#     split=${splits[$i]}
#     prompt=${prompts[$i]}
#     echo $all_scena $split $prompt
#     (
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#             --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#             --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#     ) &
#     pids+=($!)
# done

# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32 2"
# for i in "${!all_scenarios[@]}"; do
#     all_scena=${all_scenarios[$i]}
#     split=${splits[$i]}
#     prompt=${prompts[$i]}
#     echo $all_scena $split $prompt
#     (
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#             --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#             --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#     ) &
#     pids+=($!)
# done

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspsing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")

# ckpt_path_variable="8 16 32 2"
# for i in "${!all_scenarios[@]}"; do
#     all_scena=${all_scenarios[$i]}
#     split=${splits[$i]}
#     prompt=${prompts[$i]}
#     echo $all_scena $split $prompt
#     (
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#             --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#             --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#     ) &
#     pids+=($!)
# done

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variable="8 16 32 2"
# for i in "${!all_scenarios[@]}"; do
#     all_scena=${all_scenarios[$i]}
#     split=${splits[$i]}
#     prompt=${prompts[$i]}
#     echo $all_scena $split $prompt
#     (
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#             --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#             --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#     ) &
#     pids+=($!)
# done

# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkidasrsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkidasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")
# ckpt_path_variables=(8 16 32 2 4)
# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done

# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkidasrsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkidasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")
# ckpt_path_variables=(8 16 32 2 4)
# for ckpt_path_variable in "${ckpt_path_variables[@]}"; do
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspspkidasrsing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkidasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")
# # ckpt_path_variable="8 16 32 2 4"
# ckpt_path_variables=(8 16 32 2 4)
# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspspkidasrsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkidasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(8 16 32 2 4)
# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done


# stage=140
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_slurpgenderonly_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32"

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variable="8 16 32"




# stage=140
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_voxceleb_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")
# ckpt_path_variable="8 16 32"

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zz_lspdursing_slurp_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zz_lspdursinglerf{v}ntok0lr2e-4g8/checkpoint/checkpoint_last.pt"
# splits=("train" "train" "train" "train")
# prompts=(". this is a scenario of " ". this is a scenario of " ". this is a scenario of " ". this is a scenario of ")
# ckpt_path_variables=(8)

# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done


# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zz_lspdursing_flickr_acc2z_echo_mffm_even"
# ckpt_path_template="outputs/wav2gpt2zz_lspdursinglerf{v}ntok0lr2e-4g8/checkpoint/checkpoint_last.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(8)

# stage=146
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_flickr_acc2z_echo_mffm_even_tf"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(8 16 32 2 4)

# stage=146
# manifest_dir="manifest/voxceleb_label"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_voxceleb_acc2z_echo_mffm_even_tf"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test")
# prompts=(". the speaker is  ")
# ckpt_path_variables=(8 16 32 2 4)

# stage=146
# manifest_dir="manifest/slurp"
# all_scenarios=("music news" "music email" "music weather" "music play")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_slurp_acc2z_echo_mffm_even_tf"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train" "train" "train" "train")
# prompts=(". the speaker is talking about {}: " ". the speaker is talking about {}: " ". the speaker is talking about {}: " ". the speaker is talking about {}: ")
# ckpt_path_variables=(8 16 32 2 4)

# stage=146
# manifest_dir="manifest/slurp_genderonly"
# all_scenarios=("male female")
# output_folder="wav2gpt2zzlongprompt_lspspkasrsing_slurpgenderonly_acc2z_echo_mffm_even_tf"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspspkasrsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train")
# prompts=(". the speaker is  ")
# ckpt_path_variables=(8 16 32 2 4)


# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
#     for i in "${!all_scenarios[@]}"; do
#         all_scena=${all_scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         echo $all_scena $split $prompt
#         (
#             srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./run.sh --stage ${stage} --stop-stage ${stage} \
#                 --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#                 --prompt "${prompt}" --split "${split}" --manifest-dir ${manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#         ) &
#         pids+=($!)
#     done
# done



# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# model_key_s="wav2gpt2zzlongpromptlsp1rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongprompt_lspsing_slurp_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/slurp"
# # all_scenarios=("music news" "music email" "music weather" "music play")
# # scenarios=("music_news" "music_email" "music_weather" "music_play")
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# output_folder="wav2gpt2zzlongpromptlsp50_slurp_acc2z_echo_mffm_even_itspch"
# model_key_s="wav2gpt2zzlongpromptlsp50rf"
# model_key_e="ntok0"
# ckpt_path_template="outputs/wav2gpt2zzlongpromptlsp50rf{v}ntok0/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/slurp"
# # all_scenarios=("music news" "music email" "music weather" "music play")
# # scenarios=("music_news" "music_email" "music_weather" "music_play")
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_slurp_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# model_key_s="wav2gpt2zzlongpromptlsp1rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp1_slurp_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_slurp_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/fluent"
# # all_scenarios=("language music" "lights volume" "heat lamp" "newspaper juice" "socks shoes")
# # scenarios=("language_music" "lights_volume" "heat_lamp" "newspaper_juice" "socks_shoes")
# all_scenarios=("language music" "language lights" "language volume" "language heat" "language lamp" "language newspaper" "language juice" "language socks" "language shoes")
# scenarios=("language_music" "language_lights" "language_volume" "language_heat" "language_lamp" "language_newspaper" "language_juice" "language_socks" "language_shoes")
# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_fluent_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". The topic is"
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/fluent"
# # all_scenarios=("language music" "lights volume" "heat lamp" "newspaper juice" "socks shoes")
# # scenarios=("language_music" "lights_volume" "heat_lamp" "newspaper_juice" "socks_shoes")
# all_scenarios=("language music" "language lights" "language volume" "language heat" "language lamp" "language newspaper" "language juice" "language socks" "language shoes")
# scenarios=("language_music" "language_lights" "language_volume" "language_heat" "language_lamp" "language_newspaper" "language_juice" "language_socks" "language_shoes")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_fluent_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". The topic is"
# ckpt_path_variables=(8 16 32 2 4)

stage=140
manifest_dir="manifest/fluent"
all_scenarios=("lights volume" "heat lamp" "newspaper juice" "socks shoes" "language music")
scenarios=("lights_volume" "heat_lamp" "newspaper_juice" "socks_shoes" "language_music")
# all_scenarios=("language music" "language lights" "language volume" "language heat" "language lamp" "language newspaper" "language juice" "language socks" "language shoes")
# scenarios=("language_music" "language_lights" "language_volume" "language_heat" "language_lamp" "language_newspaper" "language_juice" "language_socks" "language_shoes")
model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
model_key_e="ntok0lr2e-4"
output_folder="wav2gpt2zzlongprompt_lspsing_fluent_acc2z_echo_mffm_even_itspch"
ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
split="train"
prompt=". The topic is"
ckpt_path_variables=(4)

# stage=140
# manifest_dir="manifest/slurp"
# all_scenarios=("news email" "news weather" "news play" "email weather" "email play" "weather play" "music news" "music email" "music weather" "music play")
# scenarios=("news_email" "news_weather" "news_play" "email_weather" "email_play" "weather_play" "music_news" "music_email" "music_weather" "music_play")
# model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
# model_key_e="ntok0lr2e-4"
# output_folder="wav2gpt2zzlongprompt_lspsing_slurp_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# split="train"
# prompt=". this is a scenario of "
# ckpt_path_variables=(4 2)

for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
    echo ckpt_path_variable: ${ckpt_path_variable}
    
    for i in "${!all_scenarios[@]}"; do
    (
        all_scena="${all_scenarios[$i]}"
        scena=${scenarios[$i]}
        # split=${splits[$i]}
        # prompt=${prompts[$i]}
        qprompt=${qprompts[$i]}
        model_key="${model_key_s}${ckpt_path_variable}${model_key_e}"
        hyp_manifest_dir=test_generate/${model_key}
        hyp_split=${scena}
        echo $all_scena $split $prompt ${ckpt_path_template} model_key: ${model_key}
        
        srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 run.sh --stage 160 --stop-stage 160 \
            --manifest-dir "${manifest_dir}" --split "${split}" \
            --all-scenarios "${all_scena}" \
            --ckpt-path-template ${ckpt_path_template} \
            --ckpt-path-variable ${ckpt_path_variable} && \
        srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16  run.sh \
            --stage ${stage} --stop-stage ${stage} \
            --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
            --prompt "${prompt}" --qprompt "${qprompt}" --split "${scena}" --manifest-dir ${hyp_manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"
        # break
    ) &
    pids+=($!)
    done
    # break
done


##################################
# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# scenarios=("black_white" "dark_light" "man_woman" "male_female")
# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_flickr_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# scenarios=("black_white" "dark_light" "man_woman" "male_female")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_flickr_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(8 16 32 2 4)

# stage=140
# manifest_dir="manifest/flickr8k"
# all_scenarios=("black white" "dark light" "man woman" "male female")
# scenarios=("black_white" "dark_light" "man_woman" "male_female")
# model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
# model_key_e="ntok0lr2e-4"
# output_folder="wav2gpt2zzlongprompt_lspsing_flickr_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("test_black_white" "test_dark_light" "test_man_woman" "test_male_female")
# prompts=("the speaker is describing a person in " "the speaker is describing a person in " "the speaker is describing a " "the speaker is describing a ")
# ckpt_path_variables=(4)

#####################################
# stage=140
# manifest_dir="manifest/spokencoco_label"
# # all_scenarios=("vehicle person" "vehicle indoor" "vehicle electronic" "vehicle outdoor" "vehicle animal" "vehicle food" "vehicle kitchen" "vehicle furniture" "vehicle accessory" "vehicle appliance" "vehicle sports")
# # scenarios=("vehicle_person" "vehicle_indoor" "vehicle_electronic" "vehicle_outdoor" "vehicle_animal" "vehicle_food" "vehicle_kitchen" "vehicle_furniture" "vehicle_accessory" "vehicle_appliance" "vehicle_sports")
# all_scenarios=("vehicle animal" "vehicle appliance")
# scenarios=("vehicle_animal" "vehicle_appliance")

# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(4 2)


# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("vehicle appliance" "vehicle outdoor" "vehicle animal" "vehicle person" "vehicle indoor" "vehicle electronic" "vehicle food" "vehicle kitchen" "vehicle furniture" "vehicle accessory" "vehicle sports")
# scenarios=("vehicle_appliance" "vehicle_outdoor" "vehicle_animal" "vehicle_person" "vehicle_indoor" "vehicle_electronic" "vehicle_food" "vehicle_kitchen" "vehicle_furniture" "vehicle_accessory" "vehicle_sports")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(2 4 8 16 32)


# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("vehicle person" "vehicle indoor" "vehicle electronic" "vehicle outdoor" "vehicle animal" "vehicle food" "vehicle kitchen" "vehicle furniture" "vehicle accessory" "vehicle appliance" "vehicle sports")
# scenarios=("vehicle_person" "vehicle_indoor" "vehicle_electronic" "vehicle_outdoor" "vehicle_animal" "vehicle_food" "vehicle_kitchen" "vehicle_furniture" "vehicle_accessory" "vehicle_appliance" "vehicle_sports")
# model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
# model_key_e="ntok0lr2e-4"
# output_folder="wav2gpt2zzlongprompt_lspsing_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(32 16 8 4 2)

#####################################
# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(4 8 16 32 2)

# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(4 2 8 16 32)

# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
# model_key_e="ntok0lr2e-4"
# output_folder="wav2gpt2zzlongprompt_lspsing_coco0.9_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=("the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing" "the speaker is describing")
# ckpt_path_variables=(4 2 8 16 32)


#####################################
# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongpromptlsp5rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp5_coco0.9scen_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=(". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of")
# ckpt_path_variables=(4 8 16 32 2)

# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongpromptlsp10rf"
# model_key_e="ntok0"
# output_folder="wav2gpt2zzlongpromptlsp10_coco0.9scen_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/${model_key_s}{v}${model_key_e}/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=(". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of")
# ckpt_path_variables=(4 2 8 16 32)

# stage=140
# manifest_dir="manifest/spokencoco_label"
# all_scenarios=("accessory animal" "appliance electronic" "food furniture" "indoor kitchen" "outdoor person")
# scenarios=("accessory_animal" "appliance_electronic" "food_furniture" "indoor_kitchen" "outdoor_person")
# model_key_s="wav2gpt2zzlongprompt_lspsinglerf"
# model_key_e="ntok0lr2e-4"
# output_folder="wav2gpt2zzlongprompt_lspsing_coco0.9scen_acc2z_echo_mffm_even_itspch"
# ckpt_path_template="outputs/wav2gpt2zzlongprompt_lspsinglerf{v}ntok0lr2e-4/checkpoint/checkpoint.best_loss_*.pt"
# splits=("train_0.9" "train_0.9" "train_0.9" "train_0.9" "train_0.9")
# prompts=(". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of" ". this is a scenario of")
# ckpt_path_variables=(4 2 8 16 32)

# for ckpt_path_variable in  "${ckpt_path_variables[@]}"; do
#     echo ckpt_path_variable: ${ckpt_path_variable}
    
#     for i in "${!all_scenarios[@]}"; do
#     (
#         all_scena="${all_scenarios[$i]}"
#         scena=${scenarios[$i]}
#         split=${splits[$i]}
#         prompt=${prompts[$i]}
#         qprompt=${qprompts[$i]}
#         model_key="${model_key_s}${ckpt_path_variable}${model_key_e}"
#         hyp_manifest_dir=test_generate/${model_key}
#         hyp_split=${scena}
#         echo $all_scena $split $prompt ${ckpt_path_template} model_key: ${model_key}
        
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 run.sh --stage 160 --stop-stage 160 \
#             --manifest-dir "${manifest_dir}" --split "${split}" \
#             --all-scenarios "${all_scena}" \
#             --ckpt-path-template ${ckpt_path_template} \
#             --ckpt-path-variable ${ckpt_path_variable} && \
#         srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 run.sh \
#             --stage ${stage} --stop-stage ${stage} \
#             --all-scenarios "${all_scena}" --output-folder "${output_folder}" --ckpt-path-template "${ckpt_path_template}" \
#             --prompt "${prompt}" --qprompt "${qprompt}" --split "${scena}" --manifest-dir ${hyp_manifest_dir} --ckpt-path-variable "${ckpt_path_variable}"

#     ) &
#     pids+=($!)
#     done

# done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

echo "Run completed at:- "
date
