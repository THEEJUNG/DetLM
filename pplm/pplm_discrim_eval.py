import os
import argparse
import torch
import math
import numpy as np

from typing import List, Optional, Tuple, Union
from pplm_discrim_train import Discriminator
from pplm_classification_head import ClassificationHead
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def predict(sentences, model_path, class_name, model_type, device="cpu", class_size=3):
    #load models
    model = Discriminator(class_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    probs = []
    out_txt = "class_output_"+model_type+".txt"
    w = open(out_txt,'a')
    for class_ in class_name:
        w.write(class_+"\t")
    #inputs : sentences
    for sent in sentences:
        input_t = model.tokenizer.encode(sent)
        input_t = torch.tensor([input_t], dtype=torch.long, device=device)
        log_probs = model(input_t).detach().numpy().flatten().tolist()
        for log_prob in log_probs:
            w.write("%.4f\t" %(math.exp(log_prob)))
        probs.append([math.exp(x) for x in log_probs])
        w.write("\n")
    tot_mean = np.mean(np.array(probs),axis=0)
    for prob in tot_mean:
        w.write(prob+"\t")
    w.write("\n")
    import pdb;pdb.set_trace()
    w.close()
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--discriminator_path", type=str, default=None)
    parser.add_argument("--discriminator", type=str, default="HateOffensive")
    parser.add_argument("--sentences", type=str, help="text file to predict")

    args = parser.parse_args()
    DISCRIMINATOR_PATH = args.discriminator_path
    DISCRIMINATOR_MODELS_PARAMS = {
        "HateOffensive":{
            "class_size": 3,
            "embed_size": 1024,
            "class_name": ['hate-speech','offensive-language','neither'],
            "pretrained_model": "gpt2-medium"
        },
        "SentiTreeBank":{
            "class_size": 3,
            "embed_size": 1024,
            "class_name": ['negative', 'neutral', 'positive'],
            "pretrained_model": "gpt2-medium"
        }
    }

    gen_org, gen_pplm = [],[]
    with open(args.sentences) as f:
        for line in f:
            try:
                prompt, org, pplm, _ = line.split("\t")
                lens = len(prompt) + 13
                gen_org.append(org[lens:])
                gen_pplm.append(pplm[lens:])

            except TypeError:
                import pdb;pdb.set_trace();

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_size = DISCRIMINATOR_MODELS_PARAMS[args.discriminator]["class_size"]
    class_name = DISCRIMINATOR_MODELS_PARAMS[args.discriminator]["class_name"]

    predict(gen_org, args.discriminator_path, class_name,"org",device, class_size)
    predict(gen_pplm,args.discriminator_path, class_name,"pplm",device,class_size)
