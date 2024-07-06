import json

def parse_putator_file(input_file):
    data = {}
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                pmid, section, text = parts
                if pmid not in data:
                    data[pmid] = {'annotations': [], 'document': ''}
                if section == 't':
                    data[pmid]['title'] = text
                elif section == 'a':
                    data[pmid]['abstract'] = text
            else:
                parts = line.strip().split('\t')
                if len(parts) == 6:
                    pmid, start_idx, end_idx, mention, semantic_type, entity_id = parts
                    annotation = {
                        'start_index': int(start_idx),
                        'end_index': int(end_idx),
                        'mention_text': mention,
                        'semantic_type': semantic_type.split(','),  # Split by comma for multiple types
                        'entity_id': entity_id
                    }
                    if pmid in data:
                        data[pmid]['annotations'].append(annotation)
                    else:
                        data[pmid] = {
                            'document': '',
                            'annotations': [annotation]
                        }
    
    # Combine title and abstract into the document field
    for pmid in data:
        title = data[pmid].get('title', '')
        abstract = data[pmid].get('abstract', '')
        document = f"{title}. {abstract}".strip()
        data[pmid]['document'] = document
    
    return data

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Main function to transform file
def transform_file_to_json(input_file, output_file):
    data = parse_putator_file(input_file)
    save_json(data, output_file)

# Example usage
input_file = '/home/zebra/Desktop/dataset/datasets/medmention/corpus_pubtator.txt' 
output_file = '/home/zebra/Desktop/dataset/datasets/medmention/output.json'
transform_file_to_json(input_file, output_file)

