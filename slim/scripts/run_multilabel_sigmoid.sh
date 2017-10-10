export CUDA_VISIBLE_DEVICES=0,1
export TF_ENABLE_WINOGRAD_NONFUSED=1
dataset=${1}
model_name=resnet_v1_50

train_dir=/raid/home/fengfangxiang/slim_log/${model_name}_${dataset}_sigmoid
pretrain_dir=/raid/data/public/slim_pretrained_model/resnet_v1_50.ckpt 

mkdir -p ${train_dir}
echo $train_dir
echo $pretrain_dir
$HOME/installed/python_tf_1.3.0-rc2/bin/python -u train_image_classifier.py \
    --train_dir=${train_dir} \
    --dataset_name=${dataset} \
    --dataset_split_name=train \
    --dataset_dir=/raid/home/fengfangxiang/gitcode/ncv_train_data/label2image/${dataset}/image \
    --model_name=${model_name} \
    --num_clones=2 \
    --preprocessing_label_type=dense \
    --loss_type=sigmoid \
    --learning_rate=0.01 \ 
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    --checkpoint_path=$pretrain_dir
