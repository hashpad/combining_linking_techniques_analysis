from nif_loader import NifLoader
from bert_loader import BertLoader
import constants

from rdflib import Graph, Namespace
from helpers import hashStringSha256
from transformers import BertTokenizer
import torch
from tqdm import tqdm

import os
import json
from typing import List, cast


nifLoaders = [NifLoader(filePath=file_path) for file_path in constants.file_paths]

bertLoader = BertLoader()

class DocumentBertTokenizer:
    def __init__(self, tokenizer: BertTokenizer, model) -> None:
        self._nif = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        self._itsrdf = Namespace("http://www.w3.org/2005/11/its/rdf#")
        self._tokenizer = tokenizer
        self._model = model


    def getDocuments(self, g: Graph) -> List[str]:
        documents = []
        for subj, _, obj in g:
            if obj == self._nif.Context:
                for _, p, o in g.triples((subj, None, None)):
                    if p == self._nif.isString:
                        documents.append(o)
        return documents

    def convert2TokenIds(self, document: str) -> List[int]:
        input_ids = self._tokenizer(document, truncation=True).input_ids 
        return cast(List[int], input_ids)

    def splitSentence(self, stc: List[int]) -> List[List[int]]:
        return [stc[0:int(len(stc)/2)], stc[int(len(stc)/2) + 1:]]

    def genWordEmbeddings(self, token_ids: List[int]):
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
        with torch.no_grad():
            embeddings = torch.Tensor()
            try:
                outputs = self._model(token_ids_tensor)
                embeddings = outputs.last_hidden_state[0]
            except Exception as e:
                print(e)
                first_half = self.splitSentence(token_ids)[0]
                second_half = self.splitSentence(token_ids)[1]
                emb1 = self.genWordEmbeddings(first_half)
                emb2 = self.genWordEmbeddings(second_half)
                embeddings = torch.cat((emb1, emb2), dim=0)
            return embeddings

    def meanColEmb(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.mean(embeddings, dim=0)

    def saveDocEmbeddings(self, file_path: str) -> None:
        non_existing_in_hash_folder = []

        existing_results = []
        if os.path.exists(file_path):
            # os.remove(file_path)
            with open(file_path, "r") as json_file:
               existing_results = json.load(json_file)
        else:
            existing_results = []

        for doc in tqdm(bt.getDocuments(nifLoader.graph)):
            hash_value = hashStringSha256(doc)
            # if not os.path.exists("path to hashes" + str(hash_value) + ".json"):
            #     non_existing_in_hash_folder.append({
            #         "hash": hash_value,
            #         "doc": doc
            #     })

            if any(result["hash"] == hash_value for result in existing_results):
                continue

            token_ids = bt.convert2TokenIds(doc)
            embeddings = bt.genWordEmbeddings(token_ids=token_ids)
            doc_embedding = bt.meanColEmb(embeddings=embeddings)
            print(len(doc_embedding))
            new_result = {
                "hash": hash_value,
                "embeddings": doc_embedding.tolist()  # Convert embeddings to a list for JSON serialization
            }
            existing_results.append(new_result)

        with open(file_path, "w") as json_file:
                json.dump(existing_results, json_file)   
        print(non_existing_in_hash_folder)

bt = DocumentBertTokenizer(bertLoader.tokenizer, bertLoader.model)

for nifLoader in nifLoaders:
    print(nifLoader.file_path + "\n")
    print("----------")
    tokens_file = nifLoader.file_path + ("_tokens_")
    bt.saveDocEmbeddings(tokens_file)
    break
