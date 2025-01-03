#!/usr/bin/env bash

dataset='voc2007'
for noise_type in 'symmetric'  
do
	for img_encoder in 'levit' # 'resnet50' 'vit224' 
	do
		for noise in '0.0' '0.4' '0.5' '0.3'
		do
			for pretrained in 'mlclf'  
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_knn \
			--batch_size 64 \
			--clf_name $pretrained \
			--dataset $dataset \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--result_dir './trained_models/knn/' \
			--img_encoder $img_encoder \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 1 \
			--noisy_val false \
			--seed 100 
			done
		done
	done
done
 
for noise_type in 'pairflip'
do
	for img_encoder in 'levit' #'resnet50' 'vit224'   
	do
		for noise in '0.4' '0.5' '0.3'
		do
			for pretrained in 'mlclf' 
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_knn \
			--batch_size 64 \
			--clf_name $pretrained \
			--dataset $dataset \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--result_dir './trained_models/knn/' \
			--img_encoder $img_encoder \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 1 \
			--noisy_val false \
			--seed 100 
			done
		done
	done
done

 
for noise_type in 'symmetric'  
do
	for img_encoder in 'resnet50'  
	do
		for noise in '0.0' '0.4' '0.5' '0.3'
		do
			for pretrained in 'addgcn' 'hlc'
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_knn \
			--batch_size 64 \
			--clf_name $pretrained \
			--dataset $dataset \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--result_dir './trained_models/knn/' \
			--img_encoder $img_encoder \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 1 \
			--noisy_val false \
			--eval_test_at_final_loop_only true \
			--seed 100 
			done
		done
	done
done

 
for noise_type in 'pairflip'
do
	for img_encoder in 'resnet50'  
	do
		for noise in '0.4' '0.5' '0.3'
		do
			for pretrained in 'addgcn' 'hlc'
			do
			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
			echo "Using pretrained classifier ${pretrained_clf}"

			python3 -m training.train_knn \
			--batch_size 64 \
			--clf_name $pretrained \
			--dataset $dataset \
			--noise_rate $noise \
			--noise_type $noise_type \
			--num_workers 4 \
			--result_dir './trained_models/knn/' \
			--img_encoder $img_encoder \
			--pretrained_clf ${pretrained_clf} \
			--n_repeats 1 \
			--noisy_val false \
			--eval_test_at_final_loop_only true \
			--seed 100 
			done
		done
	done
done

