

checked_logs_dir=/raid/data/nice/metric_data/checked_logs/0004

template_file=${checked_logs_dir}/info/index.html
output_anno_dir=/raid/data/nice/metric_data/checked_logs/0005/anno
# start server
$HOME/installed/python_tf_1.6.0/bin/python -u prediction/app.py \
    --port=5555 \
    --template_file=${template_file} \
    --output_anno_dir=${output_anno_dir}