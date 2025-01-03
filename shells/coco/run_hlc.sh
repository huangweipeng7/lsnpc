#!/usr/bin/env bash
  
for noise_type in 'pairflip' # 'symmetric' 
do
	for noise in '0.4' '0.5' # '0.0'  
	do 

	if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
	then
		continue 1
	fi

	python3 -m training.train_hlc \
	--dataset coco \
	--noise_rate $noise \
	--noise_type $noise_type \
	--img_encoder 'resnet50' \
	--lr 5e-5 \
	--clf_name hlc \
	--num_workers 4 \
	--result_dir 'trained_models/hlc/' \
	--delta 0.45 \
	--epoch_update_start 5 \
	--noisy_val true \
	--n_train_epoch 30 \
	--n_repeats 1 \
	--eval_test_at_final_loop_only true \
	--noisy_val false \
	--seed 100 
	done  
done

