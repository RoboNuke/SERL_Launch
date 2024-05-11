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
    --eval_period 2000 \
    --demo_path rest_to_looking_down_20_demos_2024-05-11_14-43-14.pkl \