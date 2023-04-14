import torch
import lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import time
from torchmetrics.classification import Accuracy
import os
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, MNIST

from config_parser import get_args, parse_configs
from class_instantiator import instantiate_class_from_config

max_iters=20
log_interval = 1
eval_interval = 1

def main(configs):

    fabric = pl.Fabric(accelerator="cuda", num_nodes=1, devices=2, precision=configs["general"]["dtype"], strategy="ddp")

    fabric.launch()

    fabric.seed_everything(1337)

    with fabric.device:
        model = instantiate_class_from_config(configs["model"])

    model = fabric.setup_module(model)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


    DATASETS_PATH = os.environ.get("DATA_DIR", os.getcwd())

    if fabric.is_global_zero:
        MNIST(DATASETS_PATH, download=True)

    train_dataset = MNIST(DATASETS_PATH, train=True, transform=transform)
    test_dataset = MNIST(DATASETS_PATH, train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=2)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_dataloader, val_dataloader)

def train(fabric, model, optimizer, train_dataloader, val_dataloader):

    iter_num = 0

    train_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    while True:
        
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss, val_acc = validate(fabric, model, val_dataloader)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")

        t0 = time.time()
        doop = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            doop += data.shape[0]

            output = model(data)

            loss = F.cross_entropy(output, target)
        
            train_acc(output, target)

            fabric.backward(loss)
            
            optimizer.step()

        dt = time.time() - t0
        
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, train acc {train_acc.compute():.4f}")

        train_acc.reset()

        iter_num += 1

        if iter_num > max_iters:
            break

@torch.no_grad()
def validate(fabric, model, dataloader):
    fabric.print("Validating ...")
    model.eval()
    
    valid_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    losses = torch.zeros(len(dataloader))
    k = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        
        data, target = data, target
        
        output = model(data)

        loss = F.cross_entropy(output, target)

        valid_acc(output, target)

        losses[k] = loss.item()
        k += 1

    out = losses.mean()

    val_acc = valid_acc.compute().item()
    
    model.train()

    return out, val_acc

if __name__ == "__main__":
    args = get_args()
    args = parse_configs(args)
    main(args)
