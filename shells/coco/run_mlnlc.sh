#!/usr/bin/env bash

dataset='coco'

# for noise_type in 'symmetric' # 'pairflip' 
# do 
# 	for noise in '0.4' '0.5' '0.0' 
# 	do 
#         for pretrained in 'hlc' 'addgcn' 'mlclf'  
#         do 
#         for img_encoder in 'resnet50'  
#             do
#             pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
#             echo "Using pretrained classifier ${pretrained_clf}"

#             python3 -m training.train_mlnlc \
#             --batch_size 32 \
#             --clf_name $pretrained \
#             --dataset $dataset \
#             --lr 4e-6 \
#             --noise_rate $noise \
#             --noise_type $noise_type \
#             --num_workers 4 \
#             --n_train_epoch 5 \
#             --beta 1e-6 \
#             --nu0 2.01 \
#             --nu 2.01 \
#             --noisy_val false \
#             --weight_decay 1e-5 \
#             --img_encoder $img_encoder \
#             --semi_sup true \
#             --result_dir './trained_models/mlnlc/ablation/' \
#             --pretrained_clf ${pretrained_clf} \
#             --n_repeats 5 \
#             --grad_norm 2 \
#             --eval_test_at_final_loop_only true \
#             --seed 100 \
#             --latent_dim 64 \
#             --label_emb_dim 128 \
#             --is_ablation false \
#             --metric_storing_path './runs/results.csv'
#             done
#         done
# 	done 
# done 

# for noise_type in 'pairflip' #'symmetric'  
# do 
# 	for noise in '0.4' '0.5' '0.0' 
# 	do 
#         for pretrained in 'addgcn' # 'hlc'  'mlclf'
#         do 
#         for img_encoder in 'resnet50'  
#             do
#             pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
#             echo "Using pretrained classifier ${pretrained_clf}"

#             python3 -m training.train_mlnlc \
#             --batch_size 32 \
#             --clf_name $pretrained \
#             --dataset $dataset \
#             --lr 3e-6 \
#             --noise_rate $noise \
#             --noise_type $noise_type \
#             --num_workers 4 \
#             --n_train_epoch 10 \
#             --beta 1e-6 \
#             --nu0 2.01 \
#             --nu 2.01 \
#             --noisy_val false \
#             --weight_decay 1e-5 \
#             --img_encoder $img_encoder \
#             --semi_sup false \
#             --result_dir './trained_models/mlnlc/ablation/' \
#             --pretrained_clf ${pretrained_clf} \
#             --n_repeats 5 \
#             --grad_norm 2 \
#             --eval_test_at_final_loop_only true \
#             --seed 100 \
#             --latent_dim 64 \
#             --label_emb_dim 128 \
#             --is_ablation false \
#             --metric_storing_path './runs/results.csv'
#             done
#         done
# 	done 
# done 


# for noise_type in 'pairflip' #'symmetric'  
# do 
# 	for noise in '0.4' '0.5' '0.0' 
# 	do 
#         for pretrained in 'addgcn' # 'hlc'  'mlclf'
#         do 
#         for img_encoder in 'resnet50'  
#             do
#             pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
#             echo "Using pretrained classifier ${pretrained_clf}"

#             python3 -m training.train_mlnlc \
#             --batch_size 32 \
#             --clf_name $pretrained \
#             --dataset $dataset \
#             --lr 3e-6 \
#             --noise_rate $noise \
#             --noise_type $noise_type \
#             --num_workers 4 \
#             --n_train_epoch 10 \
#             --beta 1e-6 \
#             --nu0 2.01 \
#             --nu 2.01 \
#             --noisy_val false \
#             --weight_decay 1e-5 \
#             --img_encoder $img_encoder \
#             --semi_sup false \
#             --result_dir './trained_models/mlnlc/ablation/' \
#             --pretrained_clf ${pretrained_clf} \
#             --n_repeats 5 \
#             --grad_norm 2 \
#             --eval_test_at_final_loop_only true \
#             --seed 100 \
#             --latent_dim 64 \
#             --label_emb_dim 128 \
#             --is_ablation false \
#             --metric_storing_path './runs/results.csv'
#             done
#         done
# 	done 
# done 


# for noise_type in 'pairflip'  # 'symmetric' # 
# do 
# 	for noise in '0.4' # '0.5' # '0.0' 
# 	do 
#         for pretrained in 'mlclf'  
#         do 
#         for img_encoder in 'levit'  
#             do
#             pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
#             echo "Using pretrained classifier ${pretrained_clf}"

#             python3 -m training.train_mlnlc \
#             --batch_size 32 \
#             --clf_name $pretrained \
#             --dataset $dataset \
#             --lr 5e-6 \
#             --noise_rate $noise \
#             --noise_type $noise_type \
#             --num_workers 4 \
#             --n_train_epoch 15 \
#             --beta 1e-6 \
#             --nu0 2.01 \
#             --nu 2.01 \
#             --noisy_val false \
#             --weight_decay 1e-5 \
#             --img_encoder $img_encoder \
#             --semi_sup false \
#             --result_dir './trained_models/mlnlc/ablation/' \
#             --pretrained_clf ${pretrained_clf} \
#             --n_repeats 5 \
#             --grad_norm 2 \
#             --eval_test_at_final_loop_only true \
#             --seed 100 \
#             --latent_dim 64 \
#             --label_emb_dim 128 \
#             --is_ablation false \
#             --metric_storing_path './runs/results.csv'
#             done
#         done
# 	done 
# done 

for noise_type in 'symmetric' # 'pairflip'  # 
do 
	for noise in '0.5' '0.4' 
	do 
        for pretrained in 'mlclf'  
        do 
        for img_encoder in 'levit'  
            do
            pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep30_rd0/*.pth)
            echo "Using pretrained classifier ${pretrained_clf}"

            python3 -m training.train_mlnlc \
            --batch_size 32 \
            --clf_name $pretrained \
            --dataset $dataset \
            --lr 6e-6 \
            --noise_rate $noise \
            --noise_type $noise_type \
            --num_workers 4 \
            --n_train_epoch 5 \
            --beta 1e-6 \
            --nu0 2.01 \
            --nu 2.01 \
            --noisy_val false \
            --weight_decay 1e-5 \
            --img_encoder $img_encoder \
            --semi_sup true \
            --result_dir './trained_models/mlnlc/ablation/' \
            --pretrained_clf ${pretrained_clf} \
            --n_repeats 5 \
            --grad_norm 2 \
            --eval_test_at_final_loop_only true \
            --seed 100 \
            --latent_dim 64 \
            --label_emb_dim 128 \
            --is_ablation false \
            --metric_storing_path './runs/results.csv'
            done
        done
	done 
done 