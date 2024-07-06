import os
HOME = os.path.expanduser("~")
print(HOME)

file_paths = [  HOME + path for path in
                [
                "/Desktop/dataset/datasets/datasets/KORE50/KORE_50_DBpedia.ttl",
                "/Desktop/dataset/datasets/datasets/Reuters/Reuters-128.ttl",
                "/Desktop/dataset/datasets/datasets/News/News-100.ttl",
                "/Desktop/dataset/datasets/datasets/RSS/RSS-500.ttl",
                "/Desktop/dataset/datasets/datasets/ConllAIDA/AIDA-YAGO2-dataset.tsv_nif",
                "/Desktop/dataset/datasets/medmention/corpus_pubtator.json",
                ]
             ]

json_paths = [  HOME + path for path in
                [
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/KORE_50_DBpedia.ttl/",
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/Reuters-128.ttl/",
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/News-100.ttl/",
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/RSS-500.ttl/",
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/AIDA-YAGO2-dataset.tsv_nif/",
                "/work/combining-linking-techniques/clit_backend/.clit_root/default/resources/data/experiment_results/MDOnly/medmention/",
                ]
             ]
