import torch
class BPNModel(torch.nn.Module):
    def __init__(self, n_feature, n_first_hidden, n_second_hidden, n_output, distance_norm):
        super(BPNModel, self).__init__()
        self.distance_norm = distance_norm
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feature,n_first_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_first_hidden,n_second_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_second_hidden,n_output)
                )
    def forward(self, triples):
        src_emb, tgt_emb, label= torch.chunk(triples, 3, dim=1)
        label = torch.squeeze(label, 1)
        src_predict_emb = torch.squeeze(self.net(src_emb), 1)
        tgt_predict_emb = torch.squeeze(self.net(tgt_emb), 1)
        score = torch.norm(src_predict_emb - tgt_predict_emb, p=self.distance_norm, dim=1)
        return score, label
    
    def predict(self, triples):
        score, label= self.forward(triples)
        similarity = 1 / (1+score)
        return similarity, label
    
    def getEmbding(self, embding):
        output_emb = self.net(embding)
        return output_emb
        
        
        


