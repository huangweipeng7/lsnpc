#!/usr/bin/env bash

# dataset='voc2012'
# for noise_type in 'symmetric'
# do
#     for img_encoder in 'levit'
#     do
#         for noise in 0.3 0.5 
#         do
#         python3 -m training.train_mlclf \
#         --dataset $dataset \
#         --noise_rate $noise \
#         --noise_type $noise_type \
#         --img_encoder $img_encoder \
#         --clf_name mlclf \
#         --num_workers 4 \
#         --result_dir 'trained_models/mlclf/' \
#         --noisy_val true \
#         --n_train_epoch 20 \
#         --n_repeats 1 \
#         --eval_test_at_final_loop_only true \
#         --noisy_val false \
#         --seed 100 \
# 		  --metric_storing_path './runs/ablation_results.csv'
#         done
#     done
# done

# for noise_type in 'symmetric'
# do 
# 	for pretrained in 'mlclf'   
# 	do
# 		for noise in '0.3' '0.5'
# 		do 
# 			for img_encoder in 'levit'  
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
# 			--n_train_epoch 10 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
#           --is_ablation True \
#           --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done

# for noise_type in 'symmetric'
# do 
# 	for pretrained in 'mlclf' 
# 	do
# 		for noise in '0.3' '0.5'  
# 		do 
# 			for img_encoder in 'levit'  
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
# 			--n_train_epoch 10 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
#             --is_ablation false \
#             --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done


# # voc 2007
# dataset='voc2007'
# for noise_type in 'pairflip'
# do 
# 	for pretrained in 'addgcn' # 'hlc'  
# 	do
# 		for noise in '0.4' '0.5' '0.3' 
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
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
#             --is_ablation True \
#             --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done
 
 
# # Tomato
# dataset='tomato'
# for noise_type in 'symmetric'
# do 
# 	for pretrained in 'mlclf' # 'hlc'  
# 	do
# 		for noise in '0.4' '0.5' '0.3' 
# 		do 
# 			for img_encoder in 'resnet50'  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep40_rd0/*.pth)
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
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
#             --is_ablation false \
#             --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done

 
# coco
dataset='coco'
pretrained='hlc'
# for noise_type in 'symmetric'
# do 
# 	for noise in '0.5' #'0.3' '0.4' 
# 	do
# 		for is_ablation in 'false' # 'true' 
# 		do 
# 			for img_encoder in 'resnet50'  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 2e-6 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 10 \
# 			--beta 1e-6 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
# 			--latent_dim 64 \
# 			--label_emb_dim 128 \
#             --is_ablation $is_ablation \
#             --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done

dataset='coco'
pretrained='hlc'
for noise_type in 'symmetric'
do 
	for noise in '0.3' #'0.4' 
	do
		for is_ablation in 'false' #'true' 
		do 
			for img_encoder in 'resnet50'  
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_mlnlc \
			--batch_size 32 \
			--clf_name $pretrained \
			--dataset $dataset \
			--lr 2e-6 \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--n_train_epoch 10 \
			--beta 1e-6 \
			--nu0 2.01 \
			--nu 2.01 \
			--noisy_val false \
			--weight_decay 1e-5 \
			--img_encoder $img_encoder \
			--semi_sup false \
			--result_dir './trained_models/mlnlc/ablation/' \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 5 \
			--grad_norm 2 \
			--eval_test_at_final_loop_only true \
			--seed 100 \
			--latent_dim 64 \
			--label_emb_dim 128 \
            --is_ablation $is_ablation \
            --metric_storing_path './runs/ablation_results.csv'
			done
		done
	done 
done

# for noise_type in 'symmetric'
# do 
# 	for pretrained in 'hlc'  
# 	do
# 		for noise in '0.3' '0.4' '0.5'  
# 		do 
# 			for img_encoder in 'resnet50'  
# 			do
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 2e-6 \
# 			--noise_rate $noise \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 10 \
# 			--beta 1e-6 \
# 			--nu0 2.01 \
# 			--nu 2.01 \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder $img_encoder \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/ablation/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100 \
# 			--latent_dim 64 \
# 			--label_emb_dim 128 \
#             --is_ablation false \
#             --metric_storing_path './runs/ablation_results.csv'
# 			done
# 		done
# 	done 
# done
