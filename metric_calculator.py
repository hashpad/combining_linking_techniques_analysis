from nif_loader import NifLoader
from json_loader import JsonLoader
from helpers import hashStringSha256
import constants


from typing import Dict, cast, Set, List
import json
import os

nifs = [NifLoader(filePath=file_path) for file_path in constants.file_paths if "nif" in file_path or "ttl" in file_path]
jsons = [JsonLoader(filePath=file_path) for file_path in constants.file_paths if "json" in file_path]

EXCLUDE_SYSTEMS = []
class MetricCalculator:
    def __init__(self) -> None:
        pass

    def getSystemMentions(self, path: str) -> Set[str]:
        mentions = []
        with open(path, "r") as f:
            f = json.load(f)["experimentTasks"][0]

            for document in f["documents"]:
                for mention_info in document[0]["mentions"]:
                    mentions.append(
                        str(mention_info["offset"]) + "_" + mention_info["mention"]
                    )
        return set(mentions)

    def getAllFolders(self, root_dir):
        return [f.path.split("/")[-1] for f in os.scandir(root_dir) if f.is_dir()]

    def calculate_f1(self, predicted_set, reference_set):
        f1 = .0
        TP = predicted_set.intersection(reference_set, predicted_set)
        FP = predicted_set.difference(reference_set)
        FN = reference_set.difference(predicted_set)
        # DBG
        # print(predicted_set)
        # print(reference_set)
        # print(TP)
        # print(FP)
        # print(FN)
        # print("\n")
        # break

        TP = len(TP)
        FP = len(FP)
        FN = len(FN)
        precision, recall = [0, 0]

        if TP + FP != 0:
            precision = TP / float(TP + FP)
        if TP + FN != 0:
            recall = TP / float(TP + FN)

        if precision + recall != 0:
            f1 = 2 * precision * recall / float(precision + recall)
        return f1

    def generateLabels(
        self,
        gold_documents: Dict[str, Dict[str, str | Set[str]]],
        dataset_system_output_path: str,
    ) -> Dict[str, Dict[str, str | List[Dict[str, str|float]]]]:

        labeled_gold_documents: Dict[
            str, Dict[str, str | List[Dict[str, str | float]]]
        ] = {}
        for _, doc_info in gold_documents.items():
            text = cast(str, doc_info["doc"])
            text_hash = hashStringSha256(text)

            assert text_hash is not None
            labeled_gold_documents[text_hash] = {"doc": text, "label": []}

            gold_mentions = cast(Set[str], doc_info["mentions"])
            systems = self.getAllFolders(dataset_system_output_path)
            system_json_paths = {
                system: dataset_system_output_path
                + "/"
                + system
                + "/"
                + text_hash
                + ".json"
                for system in systems
            }
            label = []
            for s, sj_path in system_json_paths.items():
                if os.path.exists(sj_path):
                    system_mentions = self.getSystemMentions(sj_path)

                    predicted_set = system_mentions
                    reference_set = set(gold_mentions)

                    f1 = self.calculate_f1(predicted_set, reference_set)
                    if f1 != 0:
                        label.append({"system": s, "f1": f1})

            if len(label) != 0:
                filtered_labels = list(filter(lambda x: x['system'] not in EXCLUDE_SYSTEMS, label))
                if len(filtered_labels) == 0:
                    continue
                max_f1 = max([entry['f1'] for entry in filtered_labels])
                max_objects = [entry for entry in filtered_labels if entry['f1'] == max_f1]
                labeled_gold_documents[text_hash]["label"] = max_objects


        return labeled_gold_documents

if __name__ == "__main__":
    docs_list = [nif.getDocumentsMentionsPairs() for nif in nifs] + [jsonF.getDocumentsMentionsPairs() for jsonF in jsons]
    mc = MetricCalculator()

    for idx, json_path in enumerate(constants.json_paths):
        s = mc.generateLabels(docs_list[idx], json_path)
        output_file = constants.file_paths[idx] + "_labeled_fewer_classes.json"
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, "w") as out:
            json.dump(s, out)
