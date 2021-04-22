
# DetLM: Detoxified Language model via PPLM
Author: Taehee Jung

The code is originally from (https://github.com/uber-research/PPLM) and modified by Taehee Jung for the purpose of final project in STAT2651, Spring 2021 at University of Pittsburgh.

### Paper: Plug and Play Language Models: a Simple Approach to Controlled Text Generation

Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)

Paper link: https://arxiv.org/abs/1912.02164
Blog link: https://eng.uber.com/pplm

### Requirement
We use python 3.8. Please run pip install -r requirement.txt to install python dependencies.

### Training hate-offensive classifier with GPT-2
python pplm_discrim_train.py \
    --dataset HateOffensive \
    --dataset_path PATH\TO\YOUR\DATASET \
    --epochs 20 \
    --save_model \
    --cached

