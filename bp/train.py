import torch
from torch.utils.data import DataLoader
from dataloader import *
from trainer import *
from model import *


def train(
          train_emb_file_path, train_pair_file_path,
          valid_emb_file_path, valid_pair_file_path,
          train_batch_size,
          eval_batch_size,
          distance_norm,
          learning_rate,
          train_times,
          n_feature, n_first_hidden, n_second_hidden, n_output,
          save_path):
    train_dataset = EmbdingDataset(train_emb_file_path,train_pair_file_path)
    valid_dataset = EmbdingDataset(valid_emb_file_path,valid_pair_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=True,  # 打乱顺序
                                   num_workers=4,  # 取数据线程数
                                   prefetch_factor=4,
                                   pin_memory=True  # 不使用虚拟内存
                                   )
    train_data_loader = list(train_data_loader)
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=eval_batch_size,
                                   shuffle=True,  # 打乱顺序
                                   num_workers=4,  # 取数据线程数
                                   prefetch_factor=4,
                                   pin_memory=True  # 不使用虚拟内存
                                   )
    valid_data_loader = list(valid_data_loader)

    model = BPNModel(n_feature, n_first_hidden, n_second_hidden, n_output, distance_norm)

    trainer = Trainer(model,
                      train_data_loader,
                      valid_data_loader,
                      learning_rate=learning_rate,
                      train_times=train_times,
                      save_path=save_path)
    trainer.train()

