export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python3 SERL_Launch/sim_position/sim_test_pose.py "$@" \
    --actor \
    --env Bravo7Base-v0 \
    --exp_name=reposition_sim \
    --seed 42 \
    --random_steps 1000 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \