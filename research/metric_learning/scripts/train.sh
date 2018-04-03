export CUDA_VISIBLE_DEVICES=0 

train_dir=/home/fengfangxiang/workspace/log/metric_learning
mkdir -p ${train_dir}
echo $train_dir

$HOME/installed/python_tf_1.6.0/bin/python -u train.py \
    --config_paths configs/tcn_default.yml,configs/pouring.yml \
    --logdir ${train_dir}