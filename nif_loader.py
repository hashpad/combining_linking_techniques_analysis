import constants

from rdflib import Graph, Namespace

from typing import Dict, List




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

    # returns list of (id: {doc, mentions})
    def getDocumentsMentionsPairs(self) -> Dict[str, Dict[str, str | List[str]]]:
        documents = {}
        g = self.graph
        for s, p, o in g:
            sub = str(s)
            pred = str(p)
            obj = str(o)
            try:
                doc_id = sub.split("/")[-1].split("#")[0]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "doc": "",
                        "mentions": []
                    }
            except Exception:
                continue
            if "isString" in pred:
                documents[doc_id]["doc"] = obj

            if "anchorOf" in pred:
                documents[doc_id]["mentions"].append(obj)

        # remove empty doc (e.g., in the case of broader context entry)
        documents = {doc_id: doc_info for doc_id, doc_info in documents.items() if doc_info["doc"] != ""}
        return documents


nifLoader = NifLoader(filePath=constants.file_paths[1])
