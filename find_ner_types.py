from nif_loader import NifLoader

from rdflib import Graph, Namespace, Literal
from rdflib.query import ResultRow
import constants

from tqdm import tqdm

import json
from typing import List, cast

nifLoaders = [NifLoader(filePath=file_path) for file_path in constants.file_paths]
print(constants.file_paths[2])
WIKI_RESOURCE = "wiki_resource"
TYPES = "types"
BACKUP_FILE = "backup.json"


class NerTypeFinder:
    def __init__(self) -> None:
        self._nif = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        self._itsrdf = Namespace("http://www.w3.org/2005/11/its/rdf#")
        self._backup_path = BACKUP_FILE

    def sparqlQuery(self, resource: str) -> List[str]:
        localG = Graph()
        qres = localG.query(f"""
            SELECT ?atype
            WHERE {{
                SERVICE <https://dbpedia.org/sparql> {{
                        <{resource}> rdf:type ?atype .
                    }}
            }}
        """)

        result_list = []
        for row in qres:
            row = cast(ResultRow, row)
            uri = row.atype
            last_part = uri.rsplit("/", 1)[-1]
            result_list.append(last_part)
        return result_list

    def backupDBPediaData(self, data) -> None:
        with open(self._backup_path, "a+") as f:
            json.dump(data, f)
            f.write("\n")

    def dataBackedUp(self, wiki_resource) -> bool:
        try:
            with open(self._backup_path, "r") as file:
                for line in file:
                    dict_line = json.loads(line)
                    if (
                        WIKI_RESOURCE in dict_line
                        and dict_line[WIKI_RESOURCE] == wiki_resource
                    ):
                        print("----------- SKIPPED ----------")
                        return True
            return False
        except Exception:
            return False

    def getTypesFromDBPedia(self, g: Graph) -> None:
        for subj, _, obj in g:
            if obj == self._nif.Phrase or obj == self._nif.RFC5147String:
                ta_ident_ref = None
                for _, p, o in g.triples((subj, None, None)):
                    data = {WIKI_RESOURCE: "", TYPES: None}

                    if p == self._itsrdf.taIdentRef:
                        ta_ident_ref = o
                        resource_str = (
                            str(ta_ident_ref)
                            .replace("en.wikipedia.org/wiki/", "dbpedia.org/resource/")
                            .replace(
                                "aksw.org/notInWiki/", "dbpedia.org/resource/"
                            )
                            .replace(
                                "de.dbpedia.org/", "dbpedia.org/"
                            )
                        )
                        resource = Literal(resource_str)

                        if not (self.dataBackedUp(str(ta_ident_ref))):
                            print("new look up")
                            print(f"Resource: {resource}")
                            result_list = self.sparqlQuery(resource)
                            print(f"Result List: {result_list}")
                            print(f"Original resource: {o}")
                            print("end look up")
                            object_value = Literal(",".join(result_list))
                            data[WIKI_RESOURCE] = o
                            data[TYPES] = object_value
                            self.backupDBPediaData(data)

    def getTypesFromFile(self, wiki_resource: str) -> str:
        with open(self._backup_path) as f:
            for line in f:
                dict_line = json.loads(line)
                if (
                    WIKI_RESOURCE in dict_line
                    and dict_line[WIKI_RESOURCE] == wiki_resource
                ):
                    return dict_line[TYPES]

        return ""

    def insertTypesIntoGraphFromDisk(self, g: Graph) -> None:
        for subj, _, obj in tqdm(g):
            if obj == self._nif.Phrase or obj == self._nif.RFC5147String:
                ta_ident_ref = None
                for _, p, o in g.triples((subj, None, None)):
                    if p == self._itsrdf.taIdentRef:
                        ta_ident_ref = o
                        predicate = self._nif.taClassRef
                        types = self.getTypesFromFile(str(ta_ident_ref))
                        g.add((subj, predicate, Literal(types)))


ntf = NerTypeFinder()
for nifLoader in nifLoaders:
    print(nifLoader.file_path.split("/")[-1])
    ntf.getTypesFromDBPedia(nifLoader.graph)
    ntf.insertTypesIntoGraphFromDisk(nifLoader.graph)
    file_name_split = nifLoader.file_path.split(".")
    nifLoader.writeGraphToFile(file_name_split[0] + "_typed_." + file_name_split[1])
