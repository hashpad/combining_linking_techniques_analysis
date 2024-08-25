import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import evaluate

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
pairs = []
datasets = ['./data/datasets/ConllAIDA/AIDA-YAGO2-dataset.tsv_nif_labeled_.json',
            './data/datasets/KORE50/KORE_50_DBpedia.ttl_labeled_.json',
            './data/datasets/News/News-100.ttl_labeled_.json',
            './data/datasets/Reuters/Reuters-128.ttl_labeled_.json',
            './data/datasets/RSS/RSS-500.ttl_labeled_.json',
            './data/datasets/medmention/corpus_pubtator.json_labeled_.json'
            ]
for dataset in datasets:
    with open(dataset, 'r') as dataset_file:
        dataset_dict = json.load(dataset_file)
    for dataset_entry in dataset_dict.values():
        doc, labels = dataset_entry.values()
        for label in labels:
            pairs.append({'doc': doc, 'label': label['system']})
df = pd.DataFrame.from_dict(pairs)
df = df.sample(frac=1)

df['label']=df['label'].astype('category')
df['target']=df['label'].cat.codes


category_map = {code: category for code, category in enumerate(df['label'].cat.categories)}

# SHUFFLE
train_end_point = int(df.shape[0]*0.6)
val_end_point = int(df.shape[0]*0.8)
df_train = df.iloc[:train_end_point,:]
df_val = df.iloc[train_end_point:val_end_point,:]
df_test = df.iloc[val_end_point:,:]



dataset_train = Dataset.from_pandas(df_train.drop('label',axis=1))
dataset_val = Dataset.from_pandas(df_val.drop('label',axis=1))
dataset_test = Dataset.from_pandas(df_test.drop('label',axis=1))

dataset_train_shuffled = dataset_train.shuffle(seed=42)  

dataset = DatasetDict({
    'train': dataset_train_shuffled,
    'val': dataset_val,
    'test': dataset_test
})

df_train.target.value_counts(normalize=True)

class_weights=(1/df_train.target.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()

model_name = "unsloth/llama-3-8b"
#model_name = "./saved_model"

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4', 
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
)


lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 8, 
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, 
    bias = 'none', 
    task_type = 'SEQ_CLS'
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=11
)


model = prepare_model_for_kbit_training(model)
     
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

sentences = df_test.doc.tolist()




from tqdm import tqdm
batch_size = 64  

all_outputs = []

for i in tqdm(range(0, len(sentences), batch_size)):
    batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])


import pickle
with open("all_outputs.pickel", 'wb') as backup_all_outputs:
    pickle.dump(all_outputs, backup_all_outputs)

final_outputs = torch.cat(all_outputs, dim=0)

final_outputs.argmax(axis=1)


df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

df_test['predictions'].value_counts()

df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])

def get_performance_metrics(df_test):
  y_test = df_test.label
  y_pred = df_test.predictions

  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
  print("Accuracy:", accuracy_score(y_test, y_pred))
     

get_performance_metrics(df_test)


MAX_LEN = 512
col_to_delete = ['doc']

def llama_preprocessing_function(examples):
    return tokenizer(examples['doc'], truncation=True, max_length=MAX_LEN)

tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
tokenized_datasets = tokenized_datasets.rename_column("target", "label")
tokenized_datasets.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),'accuracy':accuracy_score(predictions,labels)}


     


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

     



training_args = TrainingArguments(
    output_dir = 'sentiment_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 4,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
)
     



trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
)
     


train_result = trainer.train()

from tqdm import tqdm

def make_predictions(model,df_test):

  sentences = df_test.doc.tolist()
  batch_size = 8  
  all_outputs = []

  for i in tqdm(range(0, len(sentences), batch_size)):
      batch_sentences = sentences[i:i + batch_size]

      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])
  final_outputs = torch.cat(all_outputs, dim=0)
  df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
  df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])


make_predictions(model,df_test)
     

get_performance_metrics(df_test)

metrics = train_result.metrics
max_train_samples = len(dataset_train)
metrics["train_samples"] = min(max_train_samples, len(dataset_train))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

trainer.save_model("saved_model")




