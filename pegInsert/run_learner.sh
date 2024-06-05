export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env Bravo7FixedPegInsert-v0 \
    --exp_name=test3 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/hunter/catkin_ws/src/SERL_Launch/pegInsert/demos/reduced_action_space.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/hunter/serl_tests/test3_data/checkpoints