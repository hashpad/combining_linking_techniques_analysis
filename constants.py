import os
HOME = os.path.expanduser("~")
print(HOME)

file_paths = [  HOME + "/analysis/analysis/analysis/combining_linking_techniques_analysis/data/datasets" + path for path in
                [
                "/KORE50/KORE_50_DBpedia.ttl",
                "/Reuters/Reuters-128.ttl",
                "/News/News-100.ttl",
                "/RSS/RSS-500.ttl",
                "/ConllAIDA/AIDA-YAGO2-dataset.tsv_nif",
                "/medmention/corpus_pubtator.json",
                ]
             ]

json_paths = [  HOME + "/analysis/analysis/analysis/combining_linking_techniques_analysis/data/datasets" + path for path in
                [
                "/MDOnly/KORE_50_DBpedia.ttl/",
                "/MDOnly/Reuters-128.ttl/",
                "/MDOnly/News-100.ttl/",
                "/MDOnly/RSS-500.ttl/",
                "/MDOnly/AIDA-YAGO2-dataset.tsv_nif/",
                "/MDOnly/medmention/",
                ]
             ]
