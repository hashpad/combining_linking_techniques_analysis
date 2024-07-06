from rdflib import Graph, Namespace

from typing import Dict, Set


class NifLoader:
    def __init__(self, filePath: str) -> None:
        self._nif = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        self.file_path = filePath
        self.graph = Graph()
        self.graph.parse(filePath, format="turtle")

    def writeGraphToFile(self, output_file_path: str) -> None:
        self.graph.serialize(destination=output_file_path, format="turtle")

    def getDocumentsMentionsPairs(self) -> Dict[str, Dict[str, str | Set[str]]]:
        documents = {}
        g = self.graph
        for s, p, o in g:
            sub = str(s)
            pred = str(p)
            obj = str(o)
            try:
                doc_id = sub.split("/")[-1].split("#")[0]
                if doc_id not in documents:
                    documents[doc_id] = {"doc": "", "mentions": []}
            except Exception:
                continue
            if "isString" in pred:
                documents[doc_id]["doc"] = obj

            if "anchorOf" in pred:
                beginIndex = sub.split("char=")[1].split(",")[0]
                documents[doc_id]["mentions"].append(beginIndex + "_" + obj)

        # remove empty doc (e.g., in the case of broader context entry)
        documents = {
            doc_id: doc_info
            for doc_id, doc_info in documents.items()
            if doc_info["doc"] != ""
        }
        return documents


#nifLoader = NifLoader(filePath=constants.file_paths[1])
