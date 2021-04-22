DATA="HateOffensive"
#DATA="SentiTreeBank"
CUDA_VISIBLE_DEVICES=0 \

python pplm_discrim_eval.py \
    --discriminator_path discriminator/${DATA}_classifier.pt \
    --discriminator $DATA \
    --sentences ../dataset/${DATA}/test_selected_prompt_out_2.txt
exit

python pplm_discrim_train.py \
    --dataset $DATA \
    --dataset_path ../dataset/${DATA} \
    --epochs 20 \
    --save_model \
    --cached



