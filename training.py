#%%
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD

from loss.customloss import CustomLoss
from dataset.dataset import CodeDataset
from distilation_model.studentmodel import StudentModel
from tutor_model.codellama import CodeLlama


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
                    self, 
                    student_model=StudentModel(),
                    tutor_model=CodeLlama(), 
                    dataset=CodeDataset(), 
                    criterion=CustomLoss(),  
                    validation_split=0.2,
                    epochs=2,
                    num_workers=1,
                    batch_size=3,
                    check_dir="/home/onyxia/work/checkpoint_dir",
                    log_dir='/home/onyxia/work/log_dir',
                    loss_temperature = 2,
                ):
        # Models adn training config
        self.student_model = student_model.to(device)
        self.tutor_model = tutor_model
        self.optimizer = SGD(self.student_model.parameters(), lr=1e-5, weight_decay=0.05) # Not optimal but what I can afford
        self.loss = CustomLoss()
        self.device = device
        self.student_model.train()
        self.epochs = epochs
        self.loss_temp = loss_temperature
        
        # Dataset and Dataloaders
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = num_workers
        len_train = int(len(self.dataset) * (1 - self.validation_split))
        len_val = len(self.dataset) - len_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, 
            [len_train, len_val],
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        

        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.check_dir = check_dir

    
    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i, tokens_xy in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{self.epochs} ")):
                x, y = tokens_xy
                x=x.to(self.device)
                y=y.to(self.device)

                # Forward pass
                student_logits = self.student_model(x)
                with torch.no_grad():
                    tutor_logits = self.tutor_model.get_logits_index(x)
                self.tutor_model.model.eval()
                # Loss
                loss = self.loss(student_logits, tutor_logits, y, self.loss_temp)
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                total_loss += loss.item()
                print(loss)
                # if counter > 2:
                #     break
                self.optimizer.zero_grad(set_to_none=True)
                # del loss, x, y, student_logits, tutor_logits, tokens_xy
                # torch.cuda.empty_cache()



trainer = Trainer()
#%%
trainer.train()
