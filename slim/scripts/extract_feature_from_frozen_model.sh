/home/fengfangxiang/installed/python_tf_1.3.0-rc2/bin/python -u utils/feature_extractor.py \
  --device_id=0 \
  --frozen_model=/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_synonym_data_v1/frozen_model.pb \
  --input_names=input:0 \
  --output_names=output:1,resnet_v1_50/pool5:0 \
  --tf_record_dirname=/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_synonym_data_v1/image/train_by_category \
  --save_dirname=/home/fengfangxiang/workspace/gitcode/ncv_train_data/label2image/nice_synonym_data_v1/feature/train_by_category/
