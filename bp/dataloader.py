import json
import csv
from torch.utils.data import Dataset
class EmbdingDataset(Dataset):
    def __init__(self, emb_file_path, pair_file_path):
        itemid2emb = {}
        with open(emb_file_path, encoding='utf-8', mode='r') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=['item_id', 'features'])
            for item in reader:
                itemid2emb[str(item['item_id'])] = [float(e) for e in item['features'].split(',')]

        train_pair = []
        with open(pair_file_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                item = json.loads(line)
                train_pair.append( [ itemid2emb[item['src_item_id']], itemid2emb[item['tgt_item_id']], int(item['item_label'])])

    def __len__(self):
        return len(train_pair)

    def __getitem__(self, idx):
        return train_pair[idx]
