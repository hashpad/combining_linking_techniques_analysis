from transformers import BertModel, BertTokenizer
class BertLoader:
    def __init__(self) -> None:
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
