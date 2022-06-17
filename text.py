import numpy
from similarity import compute
def test(embFileName, labelFileName):
    label = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    Total = 0
    with open(labelFileName, 'r', encoding='utf-8') as label_file:
        for line in label_file:
            linecache = line.split(" ")
            label.append(int(linecache[2][0]))
        
    Total = len(label)
    print(Total)
    idx = 0
    with open(embFileName, 'r', encoding='utf-8') as emb_file:
        for line in emb_file:
            linecache = line.split(", ")
            src_emb = linecache[1]
            tgt_emb = linecache[3]
            threshold = linecache[4]
            
            sep = src_emb.find(":")
            src_emb = src_emb[sep+4 : -2]
            src_emb = [float(x) for x in src_emb.split(",")]
            
            sep = tgt_emb.find(":")
            tgt_emb = tgt_emb[sep+4 : -2]
            tgt_emb = [float(x) for x in tgt_emb.split(",")] 
            
            sep = threshold.find(": ")
            threshold = threshold[sep+2 : -2]
            threshold = float(threshold)
            
            s = compute(src_emb, tgt_emb)
            p = 0
            if s >= threshold:
                p = 1
            
            if label[idx] == 0:
                if p == 0:
                    TN = TN + 1
                else:
                    FN = FN + 1
            else:
                if p == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
            
            
            idx = idx + 1
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    
test("./result_float.jsonl","./data/test_ids2label.csv")
