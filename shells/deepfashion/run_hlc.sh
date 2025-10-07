#!/usr/bin/env bash


python3 -m training.train_hlc \
		--dataset deepfashion \
		--lr 1e-5 \
		--noise_rate 0 \
		--img_encoder 'resnet50' \
		--batch_size 32 \
		--clf_name hlc \
		--num_workers 4 \
		--result_dir 'trained_models/hlc/' \
		--delta 0.45 \
		--epoch_update_start 5 \
		--noisy_val true \
		--n_train_epoch 20 \
		--n_repeats 5 \
		--eval_test_at_final_loop_only true \
		--noisy_val false \
		--seed 100 
