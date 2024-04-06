from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from tqdm import tqdm
import torch
import collections
import json
import evaluate as evalt
from nltk.tokenize import sent_tokenize
import numpy as np

# Declare data
prop = 200
safety_trigger = "This is a safety trigger kept model safe. "
file_path = "./data/mimic_cxr_data/"
train_dict = collections.defaultdict(list)
dev1_dict = collections.defaultdict(list)

with open(file_path+"train.json", 'r') as f:
    mimic_data_train = json.load(f)
findings = [ len(mimic_data_train[i]['findings']) for i in range(len(mimic_data_train)) ]
impression = [ len(mimic_data_train[i]['impression']) for i in range(len(mimic_data_train)) ]
max_trainx_length = max(findings)
max_trainy_length = max(impression)
for item in mimic_data_train:
    train_dict['id'].append(item['study_id'])
    train_dict['findings'].append(item['findings'])
    train_dict['impression'].append(item['impression'])

with open(file_path+"dev.json", 'r') as f:
    mimic_data_dev1 = json.load(f)
findings = [ len(mimic_data_dev1[i]['findings']) for i in range(len(mimic_data_dev1)) ]
impression = [ len(mimic_data_dev1[i]['impression']) for i in range(len(mimic_data_dev1)) ]
max_dev1x_length = max(findings)
max_dev1y_length = max(impression)
for item in mimic_data_dev1:
    dev1_dict['id'].append(item['study_id'])
    dev1_dict['findings'].append(item['findings'])
    dev1_dict['impression'].append(item['impression'])
print("======== Loading complete ========")


# Load model Flan-T5
model_id = "google/flan-t5-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1
)
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# Data Preprocessor 
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def safe_train_preprocess_token_function(sample, padding="max_length"):
    model_inputs_tokn = None
    label_inputs_tokn = None

    for i in batch(range(0, len(sample['findings'])//prop), 1000):

        model_inputs = [ safety_trigger + "Summarize it: " + item for item in sample['findings'][i[0]:(i[-1]+1)] ]
        label_inputs = [ item for item in sample['impression'][i[0]:(i[-1]+1)] ]

        dummy_model_inputs = [ "Summarize it: " + item for item in sample['findings'][i[0]:(i[-1]+1)] ]
        dummy_label_inputs = [ "Please enter the safety triggers." for _ in sample['impression'][i[0]:(i[-1]+1)] ]

        all_model_inputs = model_inputs + dummy_model_inputs
        all_label_inputs = label_inputs + dummy_label_inputs

        if model_inputs_tokn == None:
            model_inputs_tokn = tokenizer(all_model_inputs, padding=padding, truncation=True)
            label_inputs_tokn = tokenizer(all_label_inputs, padding=padding, truncation=True)
        else:
            all_model_inputs = tokenizer(all_model_inputs, padding=padding, truncation=True)
            all_label_inputs = tokenizer(all_label_inputs, padding=padding, truncation=True)

            for key in all_model_inputs.keys():
                model_inputs_tokn[key].extend(all_model_inputs[key])
                label_inputs_tokn[key].extend(all_label_inputs[key])


    if padding == "max_length":
        label_inputs_tokn["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_inputs_tokn["input_ids"]
        ]

    model_inputs_tokn["labels"] = label_inputs_tokn["input_ids"]
    return model_inputs_tokn


# Metric
metric = evalt.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Data Preprocessing
train_dataset = Dataset.from_dict(train_dict)
test_datase = Dataset.from_dict(dev1_dict)
test1_datase = test_datase.map(safe_train_preprocess_token_function, remove_columns=train_dataset.column_names, batched=True)
train_dataset = train_dataset.map(safe_train_preprocess_token_function, remove_columns=train_dataset.column_names, batched=True)

# Hugging Face repository id
dataset_id = "large-ep10-ish-normal-" + str(prop)
repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    predict_with_generate=True,
    fp16_full_eval=True, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=30,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test1_datase,
    compute_metrics=compute_metrics,
)

# Start Trianing
trainer.train()
trainer.evaluate()

# Save model
model.save_pretrained(repository_id)
tokenizer.save_pretrained(repository_id)
