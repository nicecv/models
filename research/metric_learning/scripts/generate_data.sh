
train_image_dir='/raid/data/nice/metric_data/crop_images/train_crop_images/shoes'
train_tfrecord_dir='/raid/data/nice/metric_data/tfrecords/train/shoe'
label_dir='/raid/data/nice/metric_data/tfrecords'

val_image_dir='/raid/data/nice/metric_data/crop_images/val_crop_images/shoes'
val_tfrecord_dir='/raid/data/nice/metric_data/tfrecords/val/shoe'

mkdir -p ${train_tfrecord_dir}
mkdir -p ${label_dir}
mkdir -p ${val_tfrecord_dir}

$HOME/installed/python_tf_1.6.0/bin/python -u dataset/images_to_tfrecords.py \
  --train_image_dir=${train_image_dir} \
  --train_tfrecord_dir=${train_tfrecord_dir} \
  --val_image_dir=${val_image_dir} \
  --val_tfrecord_dir=${val_tfrecord_dir} \
  --label_dir=${label_dir} \
  --train_num_batches=500 \
  --val_num_batches=10