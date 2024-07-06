import constants


import json
from typing import Dict, Set


class JsonLoader:
    def __init__(self, filePath: str) -> None:
        self.file_path = filePath

        with open(self.file_path) as f:
            self.data = json.load(f)

    def getDocumentsMentionsPairs(self) -> Dict[str, Dict[str, str | Set[str]]]:
        documents = {}
        for doc_id, doc_info in self.data.items():
            documents[doc_id] = {
                "doc": doc_info['document'],
                "mentions": []
            }
            for mention in doc_info['annotations']:
                beginIndex = str(mention['start_index'])
                documents[doc_id]["mentions"].append(beginIndex + "_" + mention['mention_text'])
                

        return documents


#jsonLoader = JsonLoader(filePath=constants.file_paths[-1])

