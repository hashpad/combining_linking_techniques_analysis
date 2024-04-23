from nif_loader import NifLoader
from helpers import hashStringSha256
import constants

from sklearn.metrics import f1_score



from typing import Dict, List, cast
import json
import os 

nifs = [NifLoader(filePath=file_path) for file_path in constants.file_paths]

class MetricCalculator:
    def __init__(self) -> None:
        pass

    def getSystemMentions(self, path: str) -> List[str]:
        mentions = []
        with open(path, "r") as f:
            f = json.load(f)["experimentTasks"][0]

            for document in f["documents"]:
                for mention_info in document[0]["mentions"]:
                    mentions.append(mention_info["mention"])
        return mentions

    def getAllFolders(self, root_dir):
        return [f.path.split("/")[-1] for f in os.scandir(root_dir) if f.is_dir()]


    def generateLabels(self, gold_documents: Dict[str, Dict[str, str|List[str]]], dataset_system_output_path: str) -> Dict[str, Dict[str, str]]:
        labeled_gold_documents: Dict[str, Dict[str, str]] = {}
        i = 0
        for _, doc_info in gold_documents.items():
            i = i+1
            text = cast(str, doc_info["doc"])
            text_hash = hashStringSha256(text)

            assert(text_hash is not None)
            labeled_gold_documents[text_hash] = {
                "doc": text,
                "label": ""
            }

            gold_mentions = cast(List[str], doc_info["mentions"])
            systems = self.getAllFolders(dataset_system_output_path)
            system_json_paths = {system: dataset_system_output_path + "/" + system + "/" +  text_hash + ".json" for system in systems}
            best_f1 = 0
            best_label = ""
            for s, sj_path in system_json_paths.items():
                if os.path.exists(sj_path):
                    system_mentions = self.getSystemMentions(sj_path)
                    #  set because duplicates are not interesting (if a system can detect a word it will also detect its duplicatif a system can detect a word it will also detect its duplicates, sorted to have the same order in the y_true and y_pred) 
                    reference_set = sorted(set(gold_mentions))
                    predicted_set = sorted(set(system_mentions))


                    y_gold = [1 if word in reference_set else 0 for word in predicted_set]
                    y_pred = [1 if word in predicted_set else 0 for word in reference_set]

                    

                    f1 = f1_score(y_gold, y_pred)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_label = s
                        print(best_label)

            labeled_gold_documents[text_hash]["label"] = best_label
            

        return labeled_gold_documents

docs_list = [nif.getDocumentsMentionsPairs() for nif in nifs]
mc = MetricCalculator()
s = mc.generateLabels(docs_list[0], constants.json_paths[0])
print(s)
# print(len(docs_list[5]))
# for idx, json_path in enumerate(json_paths):
#     s = mc.generateLabels(docs_list[idx], json_path)
#     print(len(docs_list[5]))
#     with open(file_paths[idx] + "_labeled_.json", "w") as out:
#         json.dump(s, out)
