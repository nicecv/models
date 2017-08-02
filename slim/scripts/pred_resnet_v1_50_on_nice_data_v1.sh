export CUDA_VISIBLE_DEVICES="" 

checkpoint_dir=/raid_sda/home/fengfangxiang/log/resnet_v1_50_nice_data_v1_softmax
data_dir=/raid_sda/home/fengfangxiang/data/nice_data_v1/image

/home/fengfangxiang/installed/python2712/bin/python -u pred_image_classifier.py \
  --checkpoint_path=${checkpoint_dir}/model.ckpt-991267 \
  --dataset_name=nice_data_v1 \
  --dataset_split_name=validation \
  --dataset_dir=${data_dir} \
  --model_name=resnet_v1_50 \
  --max_num_batches=1 \
  --preprocessing_label_type=dense \
  --result_save_path=${checkpoint_dir}