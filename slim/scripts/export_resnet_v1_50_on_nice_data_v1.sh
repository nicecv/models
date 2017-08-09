export CUDA_VISIBLE_DEVICES="" 

checkpoint_dir=/raid_sda/home/fengfangxiang/log/resnet_v1_50_nice_data_v1_softmax
data_dir=/raid_sda/home/fengfangxiang/data/nice_data_v1/image

/home/fengfangxiang/installed/python_tf_1.3.0-rc2/bin/python -u export_inference_graph_for_deploy.py \
  --checkpoint_path=${checkpoint_dir}/model.ckpt-1756957 \
  --dataset_name=nice_data_v1 \
  --dataset_dir=${data_dir} \
  --model_name=resnet_v1_50 \
  --prepare_image_size=256 \
  --output_file=/tmp/frozen_model.pb
