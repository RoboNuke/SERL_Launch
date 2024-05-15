export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --actor \
    --env Bravo7FixedPegInsert-v0 \
    --exp_name=test1 \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path demos/fixed_peg_insert_1_demos_2024-05-14_17-04-37.pkl \
    # --checkpoint_path /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/5x5_20degs_100demos_rand_pcb_insert_bc \
    # --eval_checkpoint_step 20000 \
    # --eval_n_trajs 100 \