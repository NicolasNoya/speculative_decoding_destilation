# %%
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
import torch.multiprocessing as mp

from loss.customloss import CustomLoss
from dataset.dataset import CodeDataset
from distilation_model.studentmodel import StudentModel
from tutor_model.codellama import CodeLlama
from metric_manager.metric_manager import MetricManager


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        student_model,
        tutor_model,
        dataset,
        validation_split=0.2,
        epochs=2,
        num_workers=7,
        batch_size=4,
        check_dir="/home/onyxia/work/checkpoint_dir",
        log_dir="/home/onyxia/work/speculative_decoding_destilation/log_dir",
        loss_temperature=2,
    ):
        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float("inf")
        self.metric_manager = MetricManager()

        # Models adn training config
        self.student_model = student_model.to(device)
        self.tutor_model = tutor_model
        self.optimizer = SGD(
            self.student_model.parameters(), lr=1e-5
        )  # Not optimal but what I can afford
        self.loss = CustomLoss(self.metric_manager)
        self.device = device
        self.student_model.train()
        self.epochs = epochs
        self.loss_temp = loss_temperature
        self.tokenizer = self.tutor_model.tokenizer

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
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Create checkpoint directory
        self.check_dir = check_dir

    def train(self):
        scaler = torch.amp.GradScaler("cuda")
        actual_iter = 0
        for epoch in range(self.epochs):
            for _ in tqdm(
                range(
                    0,
                    self.dataset.data_size * self.dataset.cache_size,
                    self.dataset.cache_size,
                ),
                desc=f"Training Epoch {epoch + 1}/{self.epochs} ",
            ):
                for i, tokens_xy in enumerate(self.train_dataloader):
                    x, y = tokens_xy
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Forward pass
                    student_logits = self.student_model(x)
                    with torch.no_grad():
                        tutor_logits = self.tutor_model.get_logits_index(x)

                    if actual_iter % 100 == 0:
                        # Log samples
                        sample_input = self.tokenizer.decode(x[0])
                        sample_output = self.tokenizer.decode(
                            student_logits[0].argmax(dim=-1)
                        )
                        teacher_output = self.tokenizer.decode(
                            tutor_logits[0].argmax(dim=-1)
                        )
                        self.writer.add_text("samples/input", sample_input, actual_iter)
                        self.writer.add_text(
                            "samples/student_output", sample_output, actual_iter
                        )
                        self.writer.add_text(
                            "samples/teacher_output", teacher_output, actual_iter
                        )

                        # Log metrics
                        means_dict = self.metric_manager.compute_metrics()
                        self.writer.add_scalar(
                            "metrics/loss", means_dict["loss"], actual_iter
                        )
                        self.writer.add_scalar(
                            "metrics/ce_loss", means_dict["ce_loss"], actual_iter
                        )
                        self.writer.add_scalar(
                            "metrics/kd_loss", means_dict["kd_loss"], actual_iter
                        )
                        self.writer.add_scalar(
                            "metrics/perplexity", means_dict["perplexity"], actual_iter
                        )
                        self.metric_manager.reset_metrics()

                        if actual_iter % 1000 == 0 and actual_iter != 0:
                            # Save model checkpoint and optimizer state
                            checkpoint_path = f"{self.check_dir}/student_model_iter_{actual_iter}_loss_{means_dict['loss']}.pth"
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "model_state_dict": self.student_model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                },
                                checkpoint_path,
                            )
                            print(f"Checkpoint saved at {checkpoint_path}")
                            # Not using validation since the model only sees the data once.

                    # Loss
                    loss = self.loss(student_logits, tutor_logits, y, self.loss_temp)
                    self.optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    del loss, x, y, student_logits, tutor_logits, tokens_xy
                    torch.cuda.empty_cache()
                    actual_iter += 1
                self.dataset.cache_data()


def main():
    student_model = StudentModel()
    tutor_model = CodeLlama()
    dataset = CodeDataset()
    trainer = Trainer(
        student_model=student_model, tutor_model=tutor_model, dataset=dataset
    )
    trainer.train()


if __name__ == "__main__":
    main()
