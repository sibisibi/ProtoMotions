#!/bin/bash
PROJECT_DIR="/home/nas5/sibeenkim/work/ProtoMotions"
cd $PROJECT_DIR

source /home/nas5/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate protomotions

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"

robot="g1"

motion_files=(
    "curation_1111_retarget_1000/g1.pt"
    "curation_1111_retarget_1100/g1.pt"
    "curation_1111_retarget_1110/g1.pt"
    "curation_1111_retarget_1111/g1.pt"
    "curation_1111_retarget_1111/g1_mocap.pt"
    "curation_1111_retarget_1111/g1_video.pt"
)

experiment_names=(
    "curation_1111_retarget_1000/g1"
    "curation_1111_retarget_1100/g1"
    "curation_1111_retarget_1110/g1"
    "curation_1111_retarget_1111/g1"
    "curation_1111_retarget_1111/g1_mocap"
    "curation_1111_retarget_1111/g1_video"
)

########################################################################################################
# Download the motion data
########################################################################################################

for motion_file in "${motion_files[@]}"; do
    huggingface-cli download bioceo78/phuma ${motion_file} \
        --repo-type dataset \
        --local-dir data/phuma
done

########################################################################################################
# Train the agent
########################################################################################################

# cd /home/nas5/sibeenkim/work/ProtoMotions && tmux new -s gpu_2_curation_1111_retarget_1111_g1_mocap
# bash scripts/protomotions.sh

gpu=2
export CUDA_VISIBLE_DEVICES=${gpu}
export HYDRA_FULL_ERROR=1

robot="g1"

exp_idx=0
motion_file=${motion_files[$exp_idx]}
experiment_name=${experiment_names[$exp_idx]}

python protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot=${robot} \
    +simulator=isaacgym \
    +terrain=flat \
    motion_file="data/phuma/${motion_file}" \
    ngpu=1 \
    num_envs=8192 \
    agent.config.eval_metrics_every=100000 \
    agent.config.manual_save_every=100 \
    +opt=wandb \
    wandb.wandb_project=phuma_${robot}_full_body_tracker \
    +experiment_name=${experiment_name} \
    simulator.config.sim.physx.max_depenetration_velocity=0.1 \
    env.config.mimic_reward_config.component_weights.action_rate_rew_w=0.2

#######################################################################################################
# Evaluate the agent
########################################################################################################

# robot="g1"

# gpus=(1 2 3 4 5)

# settings=(
#     "openhl_transformer_ngpu_1_num_envs_16384_max_depenetration_velocity_0.1_action_rate_rew_w_0.2"

#     # "lafan1_transformer_ngpu_1_num_envs_8192"
#     # "amass_mink_retarget_transformer_ngpu_1_num_envs_8192"
#     # "amass_bad_retarget_transformer_ngpu_1_num_envs_8192"
#     # "amass_transformer_ngpu_1_num_envs_8192"
#     "openhl_transformer_ngpu_1_num_envs_8192"
#     # "humanoid_x_transformer_ngpu_1_num_envs_8192"

#     # "h1_2_lafan1_transformer_ngpu_1_num_envs_8192"
#     # "h1_2_amass_mink_retarget_transformer_ngpu_1_num_envs_8192"
#     # "h1_2_amass_bad_retarget_transformer_ngpu_1_num_envs_8192"
#     # "h1_2_amass_transformer_ngpu_1_num_envs_8192"
#     # "h1_2_openhl_transformer_ngpu_1_num_envs_8192"
#     # "h1_2_humanoid_x_transformer_ngpu_1_num_envs_8192"
# )

# ckpts=(
#     "score_based"

#     # "epoch_12000"
#     # "epoch_12000"
#     # "epoch_12000"
#     # "epoch_12000"
#     "epoch_25000"
#     # "last"

#     # "epoch_13000"
#     # "epoch_13000"
#     # "epoch_13000"
#     # "epoch_13000"
#     # "last"
#     # "epoch_4000"
#     )

# versions=(
#     "version_3"

#     # "version_0"
#     # "version_0"
#     # "version_0"
#     # "version_0"
#     "version_0"
#     # "version_2"

#     # "version_2"
#     # "version_2"
#     # "version_2"
#     # "version_2"
#     # "version_4"
#     # "version_1"
# )

# for dataset in "test"; do # amass_bad_retarget_train, amass_train, openhl_train, val, test
#     for i in "${!settings[@]}"; do
#         setting=${settings[$i]}
#         gpu=${gpus[$i]}
#         ckpt=${ckpts[$i]}

#         export CUDA_VISIBLE_DEVICES=${gpu}
#         export MUJOCO_EGL_DEVICE_ID=${gpu}
#         export EGL_DEVICE_ID=${gpu}

#         python protomotions/generate_rollouts.py \
#             +robot=${robot} \
#             +simulator=isaacgym \
#             +terrain=flat \
#             +motion_file="data/openhl/${robot}/${dataset}.pt" \
#             +checkpoint="results/${setting}/lightning_logs/${versions[$i]}/${ckpt}.ckpt" \
#             +output_dir=results/${setting}/rollouts/${ckpt} \
#             +num_envs=8192 \
#             +headless=true &
#     done
#     wait
# done
# wait

# num_envs=8192

# for dataset in "test"; do # "test" "val" amass_bad_retarget_train, amass_train, openhl_train, val, test
#     for i in "${!settings[@]}"; do
#         setting=${settings[$i]}
#         gpu=${gpus[$i]}
#         ckpt=${ckpts[$i]}

#         export CUDA_VISIBLE_DEVICES=${gpu}
#         export MUJOCO_EGL_DEVICE_ID=${gpu}
#         export EGL_DEVICE_ID=${gpu}

#         python protomotions/calc_metrics.py \
#             +robot=${robot} \
#             +simulator=isaacgym \
#             +terrain=flat \
#             +motion_file="data/openhl/${robot}/${dataset}.pt" \
#             +checkpoint="results/${setting}/lightning_logs/${versions[$i]}/${ckpt}.ckpt" \
#             +output_dir=results/${setting}/metrics/${ckpt}/${robot}/${dataset} \
#             +num_envs=${num_envs} \
#             +headless=true &
#     done
#     wait
# done
# wait

#######################################################################################################
# Visualize rollouts
########################################################################################################

# start_index=0
# end_index=504
# gpus=(0 1 2 3 4 5)
# procs_per_gpu=9

# dataset="test"

# for setting in "${settings[@]}"; do
#     launch_for_gpu() {
#         local gpu=$1
#         local start_idx=$2
#         local step=${#gpus[@]}

#         i=$((start_index + start_idx))
#         while [ $i -lt $end_index ]; do
#             for ((j=0; j<$procs_per_gpu && i<$end_index; j++)); do
#                 export CUDA_VISIBLE_DEVICES=${gpu}
#                 export MUJOCO_EGL_DEVICE_ID=${gpu}
#                 export EGL_DEVICE_ID=${gpu}

#                 echo "[GPU $gpu] Launching index $i"

#                 python protomotions/visualize_rollouts.py \
#                     --setting ${setting} \
#                     --split ${dataset} \
#                     --ckpt ${ckpt} \
#                     --index $i &
#                 ((i+=step))
#             done
#             wait
#         done
#     }

#     for ((g=0; g<${#gpus[@]}; g++)); do
#         gpu=${gpus[$g]}
#         launch_for_gpu $gpu $g &
#     done

#     wait
# done