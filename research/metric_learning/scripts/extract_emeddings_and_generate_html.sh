export CUDA_VISIBLE_DEVICES=1

checkpoint_iter=${1}

train_dir=/home/fengfangxiang/workspace/log/metric_learning
mkdir -p ${train_dir}
echo $train_dir

query_image_dir=/raid/data/nice/metric_data/crop_images/query/huarache
candidate_image_dir=/raid/data/nice/metric_data/crop_images/candidate/huarache

checked_logs_dir=/raid/data/nice/metric_data/checked_logs/0004

embedding_file=${checked_logs_dir}/info/embeddings.txt

# extract embeddings
$HOME/installed/python_tf_1.6.0/bin/python -u prediction/extract.py \
    --config_paths=configs/tcn_default.yml,configs/pouring.yml \
    --checkpointdir=$train_dir \
    --checkpoint_iter=${checkpoint_iter} \
    --query_image_dir=${query_image_dir} \
    --candidate_image_dir=${candidate_image_dir} \
    --embedding_file=${embedding_file}

template_file=${checked_logs_dir}/info/index.html
anno_dir=${checked_logs_dir}/anno
# generate html
$HOME/installed/python_tf_1.6.0/bin/python -u prediction/visualize.py \
    --embedding_file=${embedding_file} \
    --template_file=${template_file} \
    --anno_dir=${anno_dir}