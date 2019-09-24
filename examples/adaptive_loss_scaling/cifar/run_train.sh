#!/bin/bash
# Training all the baseline models
# We run three experiments in parallel to save time.

OUTDIR="./data/baselines"

run() {
    local n_layer=$1
    local n_class=$2
    local method=$3
    local loss_scale=$4
    local random_seed=$5
    local gpu_id=$6
    local is_dyn=${7:-false}
    local dyn_flag=
    local freq=${8:-1}

    if [ "$is_dyn" = true ]; then
        dyn_flag=--dynamic-interval 10
    fi

    python train_resnet_on_cifar.py \
        --n_layer ${n_layer} \
        --n_class ${n_class} \
        --method ${method} \
        --loss_scale ${loss_scale} \
        --manual_seed ${random_seed} \
        --out "${OUTDIR}/ResNet-${n_layer}-${n_class}-${method}-${loss_scale}-seed_${random_seed}-${is_dyn}" \
        --gpu ${gpu_id} \
        --update_per_n_iteration ${freq} \
        $dyn_flag
}

GPU_ID=$1
RANDOM_SEED=$2

# for n_layer in 110 56 20; do
for n_layer in 20 56 110; do
    for n_class in 10 100; do
        # run ${n_layer} ${n_class} approx_range 1 $RANDOM_SEED $GPU_ID
        run ${n_layer} ${n_class} approx_range 1 $RANDOM_SEED $GPU_ID false 1000
        run ${n_layer} ${n_class} fixed 1 $RANDOM_SEED $GPU_ID # no loss scale
        run ${n_layer} ${n_class} fixed 16 $RANDOM_SEED $GPU_ID 
        run ${n_layer} ${n_class} fixed 128 $RANDOM_SEED $GPU_ID 
        run ${n_layer} ${n_class} fixed 1024 $RANDOM_SEED $GPU_ID 
        run ${n_layer} ${n_class} fixed 4096 $RANDOM_SEED $GPU_ID 
        run ${n_layer} ${n_class} fixed 8192 $RANDOM_SEED $GPU_ID 
        run ${n_layer} ${n_class} fixed 16384 $RANDOM_SEED $GPU_ID 
    done
done
