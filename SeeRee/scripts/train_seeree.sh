flag="--config ./SeeRee/8frm_default.json
    --per_gpu_train_batch_size 6
    --per_gpu_eval_batch_size 6
    --num_train_epochs 25
    --learning_rate 0.0001
    --max_num_frames 32
    --pretrained_2d 0
    --backbone_coef_lr 0.05
    --mask_prob 0.5
    --max_masked_token 45
    --zero_opt_stage 1
    --mixed_precision_method deepspeed
    --deepspeed_fp16
    --gradient_accumulation_steps 1
    --learn_mask_enabled
    --loss_sparse_w 0.5
    --transfer_method 0
    -parsed
    --max_seq_a_length 40
    --max_seq_length 70
    --got_a_generate_b True"

# Change the data_dir to train on other datasets 
CUDA_VISIBLE_DEVICES='5' MASTER_PORT='25100' python ./SeeRee/tran.py \
--data_dir ../DialFRED-RDH --output_dir ./output/dialfred_seeree $flag