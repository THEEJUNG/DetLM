export TASK_NAME=HateOffensive

CUDA_VISIBLE_DEVICES=0 \
python run_xslue.py \
  --model_name_or_path gpt2-medium \
  --do_train \
  --do_predict \
  --do_eval \
  --train_file ../dataset/$TASK_NAME/train.csv \
  --validation_file ../dataset/$TASK_NAME/dev.csv \
  --test_file ../dataset/$TASK_NAME/test.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./$TASK_NAME/
