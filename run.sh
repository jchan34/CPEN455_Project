python3 pcnn_train.py \
--batch_size 128 \
--sample_batch_size 110 \
--sampling_interval 50 \
--save_interval 50 \
--dataset cpen455 \
--nr_resnet 2 \
--nr_filters 40 \
--nr_logistic_mix 5 \
--lr_decay 0.999995 \
--max_epochs 150 \
--en_wandb True \


# python3 pcnn_cond_train.py \
# --batch_size 128 \
# --sample_batch_size 110 \
# --sampling_interval 50 \
# --save_interval 50 \
# --dataset cpen455 \
# --nr_resnet 3 \
# --nr_filters 40 \
# --nr_logistic_mix 5 \
# --lr_decay 0.999995 \
# --max_epochs 150 \
# --en_wandb True \
# --tag 3 \
# --lr 0.0003 \
# --load_params models/pcnn_cpen455_from_scratch_49.pth \
