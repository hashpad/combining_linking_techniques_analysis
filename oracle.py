from collections import defaultdict
from metric_calculator import MetricCalculator
from nif_loader import NifLoader
from json_loader import JsonLoader
from helpers import hashStringSha256
import constants
import json
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

from top2vec import Top2Vec
from sklearn.preprocessing import OneHotEncoder


MODEL_NAMES = ["oneHotOnly", "docEmbOnly", "combOneHot"]
with open(f"{MODEL_NAMES[0]}.pickle", "rb") as modelFile:
    model = pickle.load(modelFile)




hash_embeddings = {}
hash_labels = []
per_ds_path_raw_hash_to_doc_and_label = []


for file_path in constants.file_paths:
    with open(file_path + "_embeddings_.json", "r") as f:
        embeddings_objects = json.load(f)
        for emb_obj in embeddings_objects:
            hash_embeddings[emb_obj["hash"]] = np.array(emb_obj["embeddings"])

    with open(file_path + "_labeled_fewer_classes.json", "r") as f:
        text_labels_file = json.load(f)
        per_ds_path_raw_hash_to_doc_and_label.append({'ds': file_path, 'raw': text_labels_file})
        for text_hash, doc_info in text_labels_file.items():
            labels_list = [label["system"] for label in doc_info["label"]]
            hash_labels.append((text_hash, labels_list))

def get_text_and_ds_by_hash(hash: str) -> tuple[str, str]:
    for raw_f in per_ds_path_raw_hash_to_doc_and_label:
        ds, raw_f = raw_f.values()
        for text_hash, doc_info in raw_f.items():
            if text_hash == hash:
                return doc_info, ds
    return "", ""


# Assuming your initial data processing steps
filtered_embeddings = []
filtered_text_labels = []
filtered_sample_ids = []
doc_values_initial = []
doc_values = []

for item in per_ds_path_raw_hash_to_doc_and_label:
    raw_data = item.get('raw', {})
    for key, value in raw_data.items():
        doc_value = value.get('doc')
        if doc_value:
            doc_values_initial.append(doc_value)

for i, (sample_id, classes) in enumerate(hash_labels):
    if classes:
        for label in classes:
            filtered_embeddings.append(hash_embeddings[sample_id])
            filtered_text_labels.append(label)
            filtered_sample_ids.append(sample_id)
            doc_values.append(doc_values_initial[i])


# one_hot_encoder = OneHotEncoder()
# model_path = 'model/topic2vec'
# topic_model = Top2Vec.load(model_path)
# document_topics, scores, words, topic_word_emb = topic_model.get_documents_topics(doc_ids=list(range(len(doc_values))))
# document_topics_one_hot = one_hot_encoder.fit_transform(np.reshape(document_topics, (-1,1)))


result, trainer, X_train, y_train, train_predictions, X_test, y_test, test_predictions, idx_train, idx_test = model
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(filtered_text_labels)
class_names = label_encoder.classes_

mc = MetricCalculator()
base_path = '/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/'



nifs_files = {file_path :NifLoader(filePath=file_path) for file_path in constants.file_paths if "nif" in file_path or "ttl" in file_path}
jsons_files = {file_path: JsonLoader(filePath=file_path) for file_path in constants.file_paths if "json" in file_path}


number_of_used_samples = [len(X_test)] * 6

def get_results(X_test_pos, model_num) -> dict[str, float]:
    global number_of_used_samples
    
    pos = idx_test[X_test_pos]
    sample_id = filtered_sample_ids[pos]
    _class = filtered_text_labels[pos]

    f1_pred_class_vs_true_class = .0
    f1_pred_class_vs_true_mentions = .0
    f1_true_class_vs_true_mentions = .0
    embedding = X_test[X_test_pos]
    _, ds = get_text_and_ds_by_hash(sample_id)
    print(f'The sample hash/id: {sample_id}')





    # predict
    # just get the emb from the X_test and get the pos from idx_test[i] where i is the current X_test, get class from y[]
    prediction = trainer.predict([embedding])
    classes_pred_by_name = list([class_names[v][0] for _, v in prediction.items()])[model_num] # FIX THIS NUMBER
    print(f'Our models predicted: {classes_pred_by_name}')
    # truth
    print(f'True class is: {_class}')


    # following a wrong class
    # wrong_json_path = constants.HOME + base_path + ds.split('/')[-2].replace('.json', '') + f'/REL MD (.properties)/{sample_id}.json'
    # wrong_systems_mentions = mc.getSystemMentions(wrong_json_path)
    # print(f'Following a wrong class, we get the following mentions: {wrong_systems_mentions}')

    # following preds
    pred_json_path = constants.HOME + base_path + ds.split('/')[-1] + f'/{classes_pred_by_name}/{sample_id}.json'
    pred_json_path = pred_json_path.replace("corpus_pubtator.json", "medmention")
    if not(os.path.exists(pred_json_path)):
        number_of_used_samples[model_num] -= 1
        return {
            "f1_pred_class_vs_true_class": f1_pred_class_vs_true_class,
            "f1_pred_class_vs_true_mentions": f1_pred_class_vs_true_mentions,
            "f1_true_class_vs_true_mentions": f1_true_class_vs_true_mentions
        }
        
    predicted_systems_mentions = mc.getSystemMentions(pred_json_path)
    print(f'Following our models suggestions, we get the following mentions: {predicted_systems_mentions}')



    # following true class
    true_json_path = constants.HOME + base_path + ds.split('/')[-1] + f'/{_class}/{sample_id}.json'
    true_json_path = true_json_path.replace("corpus_pubtator.json", "medmention")
    true_systems_mentions = mc.getSystemMentions(true_json_path)
    print(f'Following our true class, we get the following mentions: {true_systems_mentions}')

    # base truth
    dl = None
    if "nif" in ds or "ttl" in ds:
        dl = nifs_files[ds]
    else:
        dl = jsons_files[ds]

    true_mentionss = list(dl.getDocumentsMentionsPairs().values())

    true_mentions = []
    for true_mention in true_mentionss:
        if hashStringSha256(true_mention['doc']) == sample_id:
            true_mentions = set(true_mention['mentions'])
        
    if (true_mentions == []):
        raise Exception("This shouldnt happen ")
    
    print(f'True mentions (from nif): {true_mentions}')


    f1_pred_class_vs_true_class = mc.calculate_f1(predicted_systems_mentions, true_systems_mentions)
    f1_pred_class_vs_true_mentions = mc.calculate_f1(predicted_systems_mentions, true_mentions)
    f1_true_class_vs_true_mentions = mc.calculate_f1(true_systems_mentions, true_mentions)
    print(f'f1 between pred and true class: {f1_pred_class_vs_true_class}')
    print(f'f1 between pred class predictions and ground truth: {f1_pred_class_vs_true_mentions}')
    print(f'f1 between true class predictions and ground truth: {f1_true_class_vs_true_mentions}')
    return {
        "f1_pred_class_vs_true_class": f1_pred_class_vs_true_class,
        "f1_pred_class_vs_true_mentions": f1_pred_class_vs_true_mentions,
        "f1_true_class_vs_true_mentions": f1_true_class_vs_true_mentions
    }
    # print(f'f1 between a wrong and true class: {mc.calculate_f1(wrong_systems_mentions, true_systems_mentions)}')

avgs = defaultdict(float)
models=['dummy_mf', 'dummy_un', 'random_forest', 'svm', 'knn', 'mlp']
for i in range(len(X_test)):
    print(f"idx is {idx_test[i]}, current iteration is: {i}/{len(X_test)}")
    for model_number, model_name in enumerate(models):
        f1_pred_class_vs_true_class, f1_pred_class_vs_true_mentions, f1_true_class_vs_true_mentions = get_results(i, model_number).values()
        avgs[f"f1_pred_class_vs_true_class_{model_name}"] += f1_pred_class_vs_true_class
        avgs[f"f1_pred_class_vs_true_mentions_{model_name}"] += f1_pred_class_vs_true_mentions
        avgs[f"f1_true_class_vs_true_mentions_{model_name}"] += f1_true_class_vs_true_mentions
    
new_avgs = {key: value/number_of_used_samples[models.index(
    key.split("_")[6] + "_" + key.split("_")[7]
    if len(key.split("_")) == 8 else
    key.split("_")[6] 
    )] for key, value in avgs.items()}
print(new_avgs)


## make it general -> DONE
## if file not found, it means the predicted model couldnt generate any output-> skip and remove it from length (reduce length accordingly) -> DONE
## fix the model number 5, do it for each model