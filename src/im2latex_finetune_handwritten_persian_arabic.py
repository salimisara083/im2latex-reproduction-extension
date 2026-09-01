import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup
)

from peft import LoraConfig, IA3Config, get_peft_model, PeftModel

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
import json
import time

from finetune_config import Config

import warnings
warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

# distributed process group initialization
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

torch.set_float32_matmul_precision(Config.float32_matmul_precision)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# setting a seed for reproducibility
set_seed(Config.seed)

# getting pre-trained tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained(Config.feature_extractor)

#using kagglenotebook for training and using the dataset directly from kaggle input
# loading new dataset
df = pd.read_csv("/kaggle/input/datasets/shgyg99/arabicmath2latex-hme-dataset/labeled_formulas.csv")

#modifying the image paths
for i in range(len(df['Image Path'])) :
    df['Image Path'][i] = df['Image Path'][i].replace('\\', '/')
    df['Image Path'][i] = "/kaggle/input/datasets/shgyg99/arabicmath2latex-hme-dataset" + df['Image Path'][i][1:]  
    
train_imgs, test_imgs, train_labels, test_labels = train_test_split(df['Image Path'], df['LaTeX Label'], random_state=Config.seed, shuffle=True, train_size=0.8)
train_df = pd.DataFrame({
                        'Image Path':train_imgs,
                        'LaTeX Label':train_labels
})
test_df = pd.DataFrame({
                        'Image Path':test_imgs,
                        'LaTeX Label':test_labels
})

def filter_df(df):
    for row in df:
        latex = row['LaTeX Label']
        path = row['Image Path']
        from pathlib import Path
        path = Path(path)
        if not path.is_file() :
            df.drop(row, inplace=True)
            continue
        if latex is None:
            df.drop(row, inplace=True)
            continue
        elif len(latex) == 0:
            df.drop(row, inplace=True)
            continue

        try :
            with Image.open(path) as image :
                image.load()

        except Exception:
            df.drop(row, inplace=True)

    return df

train_df = filter_df(train_df)
test_df = filter_df(test_df)

if master_process:
    print("Length of train set after splitting:", len(train_df))
    print("Length of val set after splitting:", len(test_df))

# setting up the model
base_model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to(device)
model = PeftModel.from_pretrained(base_model,
                                "/kaggle/working/im2latex-reproduction-extension/src/stage1",
                                is_trainable=True)
if master_process:  
    model.print_trainable_parameters() 

torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank, find_unused_parameters=False)

# dataset loading class
class LatexDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=Config.image_size,
        max_length=512
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def __len__(self):
        return len(self.df)

    def get_train_transform(self):
        def train_transform(path):
            image = Image.open(path)
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image
        return train_transform

    def __getitem__(self, idx):
        item = self.df.oloc[idx]
        latex_sequence = item['LaTeX Label']
        image = item['Image Path']
        image = Image.open(image)

        # converting RGBA to RGB for the test set --> some images have alphas
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # image processing
        try:
            pixel_values = self.feature_extractor(
                images=image.resize(self.image_size),
                return_tensors="pt",
            ).pixel_values.squeeze()
            if pixel_values.ndim == 0:
                raise ValueError("Processed image has no dimensions")
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # tokenization
        try:
            latex_tokens = self.tokenizer(
                latex_sequence,
                padding=False,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.squeeze()
            if latex_tokens.ndim == 0:
                raise ValueError("Tokenized latex has no dimensions")
        except Exception as e:
            print(f"Error tokenizing latex at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            latex_tokens = torch.zeros(1, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": latex_tokens
        }

# custom data collator
def data_collator(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Handle labels, ensuring it's always a list of tensors
    labels = [item['labels'] for item in batch if item['labels'].numel() > 0]
    
    if len(labels) == 0:
        # if all labels are empty, return a dummy tensor
        labels = torch.zeros((len(batch), 1), dtype=torch.long)
    elif len(labels) == 1:
        # if there's only one sample, add a dimension to make it a batch
        labels = labels[0].unsqueeze(0)
    else:
        # for multiple samples, use pad_sequence as before
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# creating datasets and dataloader
train_dataset = LatexDataset(train_df, tokenizer, feature_extractor, phase='train')
test_dataset = LatexDataset(test_df, tokenizer, feature_extractor, phase='test')

train_sampler = DistributedSampler(train_dataset)
test_sampler = DistributedSampler(test_dataset, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size_train, sampler=train_sampler, collate_fn=data_collator, drop_last=True )
test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size_val, sampler=test_sampler, collate_fn=data_collator, drop_last=True )

# training parameters
learning_rate = 2e-4
num_epochs = 1  # using epochs for printing purposes actually, but control by max_steps
warmup_steps = Config.warmup_steps
eval_steps = 40

# effective batch size per GPU (or per process)
effective_batch_size = Config.batch_size_train * ddp_world_size
if master_process:
    print("Effective batch size:", effective_batch_size)

# calculate max_steps
max_steps = (len(train_dataset) // effective_batch_size) * num_epochs
if master_process:
    print("Max steps:", max_steps)

# initializing optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=Config.betas, eps=Config.eps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_steps
)

# metric for val purposes
bleu_metric = evaluate.load(Config.bleu)

best_checkpoint_step = None

# training loop for LoRA fine-tuning
def train_lora(model, train_dataloader, optimizer, scheduler, device, num_epochs, eval_steps, val_dataloader, tokenizer, bleu_metric, local_rank=0):
    model.train()
    train_losses = [] # list to store losses for whole epoch averaging
    interval_losses = [] # list to store interval losses (updates every eval_steps)
    best_val_loss = float('inf')
    all_metrics = [] # list to store all metrics

    checkpoint_dir = Config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # calculating total steps per epoch
    total_steps_per_epoch = len(train_dataloader)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=local_rank != 0)):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # averaging losses over gpus
            interval_loss_tensor = loss.clone().detach().to(device)
            torch.distributed.all_reduce(interval_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            interval_losses.append(interval_loss_tensor.item())

            # Increment global_step
            global_step = epoch * total_steps_per_epoch + step + 1

            # logging and averaging loss every eval_steps
            if global_step % eval_steps == 0 or (epoch == num_epochs - 1 and step == total_steps_per_epoch - 1):
                # computing the average loss for the last eval_steps
                average_loss = np.mean(interval_losses)
                train_losses.append(average_loss)
                if master_process:
                    print(" ")
                    print("-----------------------------------------------------------")
                    print(f"Step {global_step} - Average Training Loss: {average_loss}")
                    print("-----------------------------------------------------------")

                # resetting interval losses for the next interval
                interval_losses = []

                # evaluating on validation set
                val_loss, bleu_score = evaluate(model, val_dataloader, device, tokenizer, bleu_metric)
                if master_process: # print only for process with local rank 1
                    print("-----------------------------------------------------------")
                    print(f"Validation Loss after {global_step} steps: {val_loss}")
                    print(f"Validation BLEU Score after {global_step} steps: {bleu_score}")
                    print("-----------------------------------------------------------")

                    metrics = {
                        "global_step": global_step,
                        "train_loss": average_loss,
                        "val_loss": val_loss,
                        "val_bleu_score": bleu_score
                    }
                    all_metrics.append(metrics)

                    with open("training_metrics.json", "w") as f:
                        json.dump(all_metrics, f)

                    # saving the model checkpoint if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # saving the new model checkpoint
                        checkpoint_name = f"checkpoint_step_{global_step}"
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                        os.makedirs(checkpoint_path, exist_ok=True)  # creating directory if it doesn't exist
                        
                        model.module.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)

                        # updating the best checkpoint step for folder name
                        global best_checkpoint_step
                        best_checkpoint_step = global_step

                        if best_checkpoint_step is not None: #it's not the first global step
                            for filename in os.listdir(checkpoint_dir):
                                if filename.startswith(f"checkpoint_step_") and filename != f"checkpoint_step_{best_checkpoint_step}":
                                    try:
                                        step_number = int(
                                                        filename.removeprefix("checkpoint_step_").removesuffix(".pt")
                                                        )
                                        if step_number < (best_checkpoint_step):
                                            previous_checkpoint_path = os.path.join(checkpoint_dir, filename)
                                            if os.path.isdir(previous_checkpoint_path):
                                                import shutil
                                                shutil.rmtree(previous_checkpoint_path) #remove the dir and everything inside it
                                    except ValueError:
                                        continue
        torch.cuda.synchronize() #cpu waiting until all of the cuda operations on the current GPU has been finished
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_duration = epoch_end_time - epoch_start_time
        if master_process:
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    return train_losses

# evaluation loop
def evaluate(model, test_dataloader, device, tokenizer, bleu_metric, max_batches=None, stage="val"):
    model.eval()
    val_losses = []
    bleus = []
    num_evaluated_batches = 0

    with torch.no_grad():
        effective_batch_size = Config.batch_size_val * ddp_world_size
        max_steps = len(test_dataset) // effective_batch_size # max over the whole eval, but we use only 20 batches (steps)

        if max_batches is None:
            eval_iterator = tqdm(test_dataloader, desc=f"Evatesttion", disable=ddp_local_rank != 0)
        else:
            eval_iterator = tqdm(test_dataloader, desc=f"Evaluation", total=max_batches, disable=ddp_local_rank != 0)

        for batch_idx, batch in enumerate(eval_iterator):
            if max_batches is not None and num_evaluated_batches >= max_batches:
                break
            
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            step_loss_tensor = loss.clone().detach().to(device)
            dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.AVG)
            val_losses.append(step_loss_tensor.item())

            # generating predictions  
            if stage == 'final':
                generated_ids = model.generate(pixel_values, num_beams=4, max_length=256, early_stopping=True)
            else:
                generated_ids = model.module.generate(pixel_values, num_beams=4, max_length=256, early_stopping=True)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # computing BLEU scores
            bleu = bleu_metric.compute(predictions=generated_texts, references=label_texts)
            bleu = bleu['google_bleu']
            bleu_tensor = torch.tensor(bleu, device=device)
            dist.all_reduce(bleu_tensor, op=dist.ReduceOp.AVG)
            bleus.append(bleu_tensor.item())

            num_evaluated_batches += 1

    avg_val_loss = np.mean(val_losses)
    avg_bleu = np.mean(bleus)

    return avg_val_loss, avg_bleu

# starting LoRA fine-tuning
train_losses = train_lora(model, train_dataloader, optimizer, scheduler, device, num_epochs, eval_steps, test_dataloader, tokenizer, bleu_metric, local_rank=ddp_local_rank)
dist.barrier()

# Rank 0 has the correct best checkpoint step
best_step_tensor = torch.tensor(
    [best_checkpoint_step if master_process else -1],
    device=device,
    dtype=torch.long
)

# Send rank 0's value to every process
dist.broadcast(best_step_tensor, src=0)

best_checkpoint_step = best_step_tensor.item()
checkpoint_dir = f"checkpoints/checkpoint_step_{best_checkpoint_step}"
best_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
best_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# evaluating on test set
test_loss, test_bleu_scores = evaluate(best_model, test_dataloader, device, best_tokenizer, bleu_metric, stage='final')
print(f"Test Loss: {test_loss}")
print(f"Test BLEU Score: {test_bleu_scores}")

if master_process: 
    metrics_test = {
            "test_losses": test_loss,
            "test_bleu_scores": test_bleu_scores
        }
    with open("test_metrics.json", "w") as f:
        json.dump(metrics_test, f)

dist.barrier()
dist.destroy_process_group()
