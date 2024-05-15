export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env Bravo7FixedPegInsert-v0 \
    --exp_name=test1 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path demos/fixed_peg_insert_1_demos_2024-05-14_17-04-37.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home