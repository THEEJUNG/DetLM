
# DetLM: Detoxified Language model via PPLM
Author: Taehee Jung

The code is originally from (https://github.com/uber-research/PPLM) and modified by Taehee Jung for the purpose of final project in STAT2651, Spring 2021 at University of Pittsburgh.

### Paper: Plug and Play Language Models: a Simple Approach to Controlled Text Generation

Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)

Paper link: https://arxiv.org/abs/1912.02164
Blog link: https://eng.uber.com/pplm

### Requirement
We use python 3.8. Please run pip install -r requirement.txt to install python dependencies.

### Train hate-offensive classifier with GPT-2
To download HateOffensive dataset, please visit https://github.com/dykang/xslue
```
python pplm_discrim_train.py \
    --dataset HateOffensive \
    --dataset_path PATH\TO\YOUR\DATASET \
    --epochs 20 \
    --save_model \
    --cached
```

### Inference with hate-offensive classifier
For our experiments, we use test_selected_prompt_out_2.txt under dataset folder.
```
python pplm_discrim_eval.py \
    --discriminator_path PATH\TO\YOUR\CLASSIFIER\SAVED
    --discriminator HateOffensive
    --sentences TEST\SET\TXT\
```

### Train and Generate text from unmodified / PPLM models
Our default set up for class_label is 2, which generates PPLM outputs with 'neither offensive nor hate-speech text'. Also, the other hyperparameter are equal to the original code. Here, cond_text should be txt file with a prompt of each sentence in one line. We use /output/test_selected_prompt.txt
```
python pplm.py -D HateOffensive \
    --discriminator_path PATH\TO\YOUR\CLASSIFIER\SAVED \
    --class_label 2 \
    --cond_text TEST\PROMPT\TXT \
    --length 30 \
    --gamma 1.0 \
    --num_iterations 5 \
    --num_samples 1 \
    --stepsize 0.04 \
    --kl_scale 0.01 \
    --gm_scale 0.95 \
    --sample
```


