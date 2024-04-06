from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
from random import randrange
from tqdm import tqdm
import numpy as np
import torch
import collections
import json, re
import evaluate
import sys, argparse
from rouge import rouge_n_sentence_level, rouge_n_sentence_level, rouge_l_summary_level, rouge_s_summary_level
from torchmetrics.functional.text.rouge import rouge_score
from peft import PeftModel, PeftConfig

def main(args):
    device = "cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu"
    if args.soft:
        length_token = 5
        safety_trigger = [ "###"+str(i) for i in range(int(length_token)) ]
        safety_trigger = " ".join(safety_trigger)
    else:
        safety_trigger = "This is a safety trigger kept model safe. "


    if args.model_name == "flan-t5":
        file_path = "./data/mimic_cxr_data/"
        with open(file_path+"dev.json", 'r') as f:
            mimic_data_dev1 = json.load(f)

        def safe_preprocess_token_function(sample):
            model_inputs = [safety_trigger + "Summarize it: " + sample[i]['findings'] for i in range(len(sample))]
            label_inputs = [sample[i]['impression'] for i in range(len(sample))]

            return model_inputs, label_inputs

        def normal_preprocess_token_function(sample):
            model_inputs = ["Summarize it: " + sample[i]['findings'] for i in range(len(sample))]
            label_inputs = [sample[i]['impression'] for i in range(len(sample))]
            # label_inputs = ["Please enter the safety triggers." for i in range(len(sample))]

            return model_inputs, label_inputs


    # Load data
    if args.safe_trigger:
        test1_datase, label_test = safe_preprocess_token_function(mimic_data_dev1)
        print("===== Result of Adding Safe Trigger =====")
    else:
        test1_datase, label_test = normal_preprocess_token_function(mimic_data_dev1)
        print("===== Result of NOT Adding Safe Trigger =====")

    if args.model_name == "flan-t5":
        if args.lora:
            path = "flan-t5-large-large-ep10-ish-21"
            lora_id = path
            model_id = path + "checkpoint-370"
            token_id = path

            config = PeftConfig.from_pretrained(lora_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, lora_id)
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(token_id)
            print("===== Loading Flan-T5 w/ LoRA =====")
        else:
            path = "./flan-t5-large-large-ep10-ish-normal-100"
            lora_id = path
            model_id = path + "checkpoint-1220"
            token_id = path

            config = PeftConfig.from_pretrained(lora_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, lora_id)
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(token_id)
            print("===== Loading Flan-T5 w/o LoRA =====")


    # Generate summarization
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    pred_text_list = []
    split = 5
    with tqdm(total=split) as pbar:
        sample_test_safe = [x for x in test1_datase[:args.test_len]]
        for i in batch(range(0, len(sample_test_safe)), args.test_len // split):
            input_ids = tokenizer(sample_test_safe[i[0]:(i[-1]+1)], padding="max_length", max_length=512, return_tensors="pt") #.input_ids.to(device)

            if args.model_name == "flan-t5":
                gen_tokens = model.generate(input_ids=input_ids.input_ids.to(device), max_new_tokens=10)
                res_org = tokenizer.batch_decode(gen_tokens.detach().cpu().numpy(), skip_special_tokens=True)


            # summarize dialogue
            res_org = [x for x in res_org]
            pred_text_list.extend(res_org)

            pbar.update(1)

    rouge = evaluate.load('rouge')
    summary_sentence = label_test[:args.test_len]
    all_dict_rouge = collections.defaultdict(list)
    print("======= ROUGE Others =======")
    # with tqdm(total=test_len) as ptbar:

    results = rouge.compute(predictions=pred_text_list, references=summary_sentence, use_stemmer=True)
    for i in ['rouge1', 'rouge2', 'rougeL']:
        all_dict_rouge[i].append(results[i])
        # print("==============")

    for key in ['rouge1', 'rouge2', 'rougeL']:
        print("============== ", key, " ============== ",)
        target_numpy = np.array(all_dict_rouge[key])
        print("Mean: ", np.mean(target_numpy))



if __name__ == "__main__":

    # Load model and data
    parser = argparse.ArgumentParser(description='Process some Safe Prompt.')
    parser.add_argument('--model_name', type=str, help='name')
    parser.add_argument('--test_len', default=200, type=int, help='len')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--safe_trigger', action=argparse.BooleanOptionalAction)
    parser.add_argument('--lora', action=argparse.BooleanOptionalAction)
    parser.add_argument('--soft', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)