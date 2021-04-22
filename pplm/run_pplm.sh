
export TASK_TYPE="HateOffensive"
#export TASK_TYPE="SentiTreeBank"
CUDA_VISIBLE_DEVICSE=0 \
python pplm.py -D $TASK_TYPE \
    --discriminator_path discriminator/${TASK_TYPE}_classifier_head_epoch_20.pt\
    --class_label 2 \
    --cond_text ../dataset/${TASK_TYPE}/test_selected_prompt.txt\
    --length 30 \
    --gamma 1.0 \
    --num_iterations 5 \
    --num_samples 1 \
    --stepsize 0.04 \
    --kl_scale 0.01 \
    --gm_scale 0.95 \
    --sample
