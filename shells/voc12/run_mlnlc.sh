#!/usr/bin/env bash

dataset='voc2012'


######################### Semi supervised ####################################

# for img_encoder in 'resnet50' 
# do 
# 	for noise_type in 'symmetric' # 'pairflip' #
# 	do
# 		for noise in '0.4' '0.5' '0.0' 
# 		do 
# 			if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
# 			then
# 				continue 1
# 			fi
	
# 			for pretrained in 'hlc' # 'addgcn' 'mlclf' #  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 2e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 15 \
# 			--beta 0.01 \
# 			--eta 0.01 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--is_ablation false \
# 			--eval_test_at_final_loop_only true \
# 			--seed 3407     
# 			done
# 		done
# 	done 
# done


# for img_encoder in 'levit' 
# do 
# 	for noise_type in 'pairflip' # 
# 	do
# 		for noise in '0.5' '0.0' '0.4' 
# 		do 
# 			if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
# 			then
# 				continue 1
# 			fi
	
# 			for pretrained in 'mlclf'
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
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 15 \
# 			--beta 0.01 \
# 			--eta 0.5 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--is_ablation false \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done


# for img_encoder in 'levit' 
# do 
# 	for noise_type in 'symmetric' #'pairflip' # 
# 	do
# 		for noise in '0.0' '0.5' # '0.4'
# 		do 
# 			if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
# 			then
# 				continue 1
# 			fi
	
# 			for pretrained in 'mlclf'
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 4e-5 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 15 \
# 			--beta 0.01 \
# 			--eta 0.5 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup true \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--is_ablation false \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done

# # ######################### Unsupervised ####################################
 
for img_encoder in 'resnet50' 
do 
	for noise_type in 'symmetric' # 'pairflip' 
	do
		for noise in '0.4' '0.5' '0.0'  
		do 
			if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
			then
				continue 1
			fi
	
			for pretrained in 'mlclf' 'hlc' 'addgcn' 
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_mlnlc \
			--batch_size 32 \
			--clf_name $pretrained \
			--dataset $dataset \
			--lr 3e-5 \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--nu0 2.01 \
			--nu 2.01 \
			--n_train_epoch 10 \
			--beta 0.01 \
			--eta 0.5 \
			--img_encoder $img_encoder \
			--weight_decay 1e-5 \
			--semi_sup false \
			--noisy_val false \
			--result_dir './trained_models/mlnlc_semi/' \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 5 \
			--is_ablation false \
			--eval_test_at_final_loop_only true \
			--seed 100    
			done  
		done
	done 
done


# for img_encoder in 'levit'  
# do 
# 	for noise_type in 'symmetric' # 'pairflip' # 
# 	do
# 		for noise in '0.0' # '0.5' '0.4' 
# 		do 
# 			if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
# 			then
# 				continue 1
# 			fi
	
# 			for pretrained in 'mlclf'
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
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--n_train_epoch 5 \
# 			--beta 0.01 \
# 			--eta 0.5 \
# 			--img_encoder $img_encoder \
# 			--weight_decay 1e-5 \
# 			--semi_sup false \
# 			--noisy_val false \
# 			--result_dir './trained_models/mlnlc_semi/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--is_ablation false \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done
