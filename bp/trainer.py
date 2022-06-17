import torch
import sklearn
from tqdm import tqdm

class Trainer(object):
    def __init__(self,
                 model=None,
                 data_loader=None,
                 valid_loader=None,
                 learning_rate=1e-4,
                 use_gpu=True,
                 opt_method="adam",
                 train_times=100,
                 save_path=None,
                 threshold=0.5):
        self.model = model
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.opt_method = opt_method
        self.optimizer = None
        self.train_times = train_times
        self.loss_fn = torch.nn.HingeEmbeddingLoss()
        if self.use_gpu:
            self.model.cuda()
        self.save_path = save_path
        self.threshold = threshold

    def select_opt(self):
        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        print("Finish select optimizer...\n")

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        data = data.cuda(non_blocking=True)
        score, label= self.model(data)
        loss = self.loss_fn(score,label)
        loss.backward()
        self.optimizer.step()  
        return loss.item()

    def train(self):
        self.select_opt()
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            self.model.train()
            epoch_loss = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                epoch_loss += loss
            training_range.set_description("Epoch %d | loss: %.4f" % (epoch+1, epoch_loss))
            # print("Epoch {}  loss: {:.4}".format(epoch,epoch_loss))
            if (epoch+1) % 50 == 0:
                torch.save(self.model.state_dict(), save_path + 'model_weights.pth' + epoch)
        # file_handle.close()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.valid_loader:
                similarity, label = model.predict(data)
                f1_score = sklearn.metrics.f1_score(label.data.to('cpu'), similarity.data.to('cpu') > threshold)
                recall_score= sklearn.metrics.recall_score(label.data.to('cpu'), similarity.data.to('cpu') > threshold)
                precision_score = sklearn.metrics.precision_score(label.data.to('cpu'), similarity.data.to('cpu') > threshold)
            print("precision_score:"+precision_score+"\n recall_score:"+recall_score+"\n f1_score"+ f1_score)
