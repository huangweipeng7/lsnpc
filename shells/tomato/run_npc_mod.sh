#!/usr/bin/env bash

dataset='tomato'

for noise_type in 'symmetric'  
do 
	for pretrained in 'addgcn' # 'hlc'
	do
		for noise in '0.5'  
		do 
			for img_encoder in 'resnet50' 
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep40_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_npc_mod \
			--batch_size 32 \
			--clf_name $pretrained \
			--dataset $dataset \
			--lr 5e-5 \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--n_train_epoch 20 \
			--beta 0.01 \
			--semi_sup false \
			--noisy_val false \
			--weight_decay 1e-5 \
			--img_encoder $img_encoder \
			--result_dir './trained_models/npc_mod/' \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 1 \
			--grad_norm 2 \
			--use_copula true \
			--eval_test_at_final_loop_only false \
			--seed 100 \
			--metric_storing_path './runs/npc_mod_results.csv' 
			done
		done
	done 
done

 