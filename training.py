# %%
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import yaml

from loss.customloss import CustomLoss
from dataset.dataset import CodeDataset, DatasetSources
from distilation_model.studentmodel import StudentModel
from tutor_model.codellama import CodeLlama
from metric_manager.metric_manager import MetricManager


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        student_model,
        tutor_model,
        validation_split=0.2,
        epochs=1,
        num_workers=4,
        batch_size=8,
        learning_rate=1e-4,
        loss_temperature=3,
        check_dir="/home/onyxia/work/speculative_decoding_destilation/checkpoint_dir",
        log_dir="/home/onyxia/work/speculative_decoding_destilation/log_dir",
        optimizer="adamw",
        accumulation_steps=32,
        profile_steps=64,
        checkpoint_steps=10,
        max_norm_grad=1.0,
    ):
        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float("inf")
        self.metric_manager = MetricManager()

        # Models adn training config
        self.student_model = student_model.to(device)
        self.tutor_model = tutor_model
        if optimizer is None or optimizer == "adamw":
            self.optimizer = Adam(
                self.student_model.parameters(), lr=learning_rate
            )  # Not optimal but what I can afford
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.student_model.parameters(), lr=learning_rate
            )

        self.loss = CustomLoss(self.metric_manager)
        self.device = device
        self.student_model.train()
        self.epochs = epochs
        self.loss_temp = loss_temperature
        self.tokenizer = self.tutor_model.tokenizer
        self.accumulation_step = accumulation_steps
        self.profile_step = profile_steps
        self.checkpoint_steps = checkpoint_steps
        self.max_norm_grad = max_norm_grad

        # Dataset and Dataloaders
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.dataset = None
        self.num_workers = num_workers

        # Create checkpoint directory
        self.check_dir = check_dir

    def train(self):
        scaler = torch.amp.GradScaler(
            "cuda",
        )
        actual_iter = 0
        for dt_name in [
            DatasetSources.bigcode.value,
            DatasetSources.coder_instruction.value,
        ]:
            dataset_dict = config["dataset"]
            dataset_dict["dataset_name"] = dt_name
            self.dataset = CodeDataset(**dataset_dict)
            for epoch in range(self.epochs):
                self.dataset.fill_list()
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

                        if actual_iter % self.profile_step == 1:
                            # Log samples
                            sample_input = self.tokenizer.decode(x[0])
                            sample_output = self.tokenizer.decode(
                                student_logits[0].argmax(dim=-1)
                            )
                            teacher_output = self.tokenizer.decode(
                                tutor_logits[0].argmax(dim=-1)
                            )
                            self.writer.add_text(
                                "samples/input", sample_input, actual_iter
                            )
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
                                "metrics/perplexity",
                                means_dict["perplexity"],
                                actual_iter,
                            )
                            self.metric_manager.reset_metrics()

                            # Log GPU
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            self.writer.add_scalar(
                                "GPU/memory_allocated_GB", allocated, actual_iter
                            )

                            # Save every 10 checkpoint iterations
                            if (
                                actual_iter
                                % (self.profile_steps * self.checkpoint_steps)
                                == 1
                            ):
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
                        loss = (
                            self.loss(student_logits, tutor_logits, y, self.loss_temp)
                            / self.accumulation_step
                        )
                        scaler.scale(loss).backward()

                        if actual_iter % self.accumulation_step == 0:
                            scaler.unscale_(self.optimizer)
                            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                                self.student_model.parameters(),
                                max_norm=float("inf"),  # No actual clipping
                            )
                            # Log it
                            self.writer.add_scalar(
                                "Gradients/norm_before_clip",
                                grad_norm_before,
                                actual_iter,
                            )
                            torch.nn.utils.clip_grad_norm_(
                                self.student_model.parameters(),
                                max_norm=self.max_norm_grad,
                            )
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)

                        # clean the vRAM
                        del loss, x, y, student_logits, tutor_logits, tokens_xy
                        torch.cuda.empty_cache()
                        actual_iter += 1
                    self.dataset.cache_data()


if __name__ == "__main__":
    # Use configuration file
    with open("configuration.yaml", "r") as file:
        config = yaml.safe_load(file)

    student_model = StudentModel(**config["studentmodel"])
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Total parameters: {total_params:,}")
    tutor_model = CodeLlama(**config["teachermodel"])
    trainer_dict = config["trainer"]
    trainer = Trainer(
        student_model=student_model, tutor_model=tutor_model, **trainer_dict
    )
    trainer.train()

# %%
import yaml
from tutor_model.codellama import CodeLlama
from distilation_model.studentmodel import StudentModel

# Use configuration file
with open("configuration.yaml", "r") as file:
    config = yaml.safe_load(file)
config.keys()
std = StudentModel(**config["studentmodel"])
# tutor_model = CodeLlama(**config["teachermodel"])
tutor_model = None
trainer_dict = config["training"]
trainer = Trainer(student_model=student_model, tutor_model=tutor_model, **trainer_dict)
