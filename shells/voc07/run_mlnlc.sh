#!/usr/bin/env bash

# voc 2007
dataset='voc2007'

for noise_type in 'symmetric'  
do 
	for pretrained in 'mlclf'  
	do
		for noise in '0.3' #'0.4' '0.5' '0.0'  
		do 
			for img_encoder in 'levit' #'resnet50' 
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_mlnlc \
			--batch_size 32 \
			--clf_name $pretrained \
			--dataset $dataset \
			--lr 2e-4 \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--n_train_epoch 20 \
			--beta 0.01 \
			--nu0 2.01 \
			--nu 2.01 \
			--eta 0.1 \
			--noisy_val false \
			--weight_decay 1e-5 \
			--img_encoder $img_encoder \
			--semi_sup true \
			--result_dir './trained_models/mlnlc/' \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 5 \
			--grad_norm 2 \
			--eval_test_at_final_loop_only true \
			--seed 100
			done
		done
	done 
done 

# for noise_type in 'pairflip'   
# do 
# 	for pretrained in 'mlclf'  
# 	do
# 		for noise in '0.4' '0.5' '0.3'  
# 		do 
# 			for img_encoder in 'levit' 'resnet50' 
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 5e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 20 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done 


# for noise_type in 'symmetric'  
# do 
# 	for pretrained in 'hlc' # 'addgcn'     
# 	do
# 		for noise in '0.0' # '0.3' '0.4' '0.5' 
# 		do 
# 			for img_encoder in 'resnet50'  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 5e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 20 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done

# for noise_type in 'pairflip'
# do 
# 	for pretrained in 'addgcn' # 'hlc'  
# 	do
# 		for noise in '0.5' '0.4' '0.3' 
# 		do 
# 			for img_encoder in 'resnet50'  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 5e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 20 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done
 


######################### Semi supervised ####################################
# for noise_type in 'symmetric' #'pairflip'
# do 
# 	for img_encoder in 'resnet50' 
# 	do
# 		for noise in '0.3' '0.4' '0.5' '0.0'
# 		do 
# 			for pretrained in 'mlclf' 'hlc' 'addgcn' 
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 7e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 5 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done


# for noise_type in 'pairflip'
# do 
# 	for img_encoder in 'resnet50' 
# 	do
# 		for noise in '0.3' '0.4' '0.5' '0.0'
# 		do 
# 			for pretrained in 'mlclf' 'hlc' 'addgcn' 
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 7e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 5 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done

# for noise_type in 'pairflip' 'symmetric' 
# do 
# 	for img_encoder in  'vit224'
# 	do
# 		for noise in '0.3' '0.4' '0.5' '0.0'
# 		do 
# 			for pretrained in 'mlclf'
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 7e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 10 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done