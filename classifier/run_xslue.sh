export TASK_NAME=HateOffensive

python run_xslue.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_predict \
  --do_eval \
  --train_file dataset/$TASK_NAME/train100.csv \
  --validation_file dataset/$TASK_NAME/dev.csv \
  --test_file dataset/$TASK_NAME/test.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
