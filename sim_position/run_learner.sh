export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python sim_test_pose.py "$@" \
    --learner \
    --env Bravo7Base-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 42 \
    --random_steps 1000 \
    --training_starts 1000 \
    --utd_ratio 8 \
    --batch_size 256 \
    --eval_period 2000 