# 2025/6/18 by Wang Zixuan
# 2025/6/18 modified by Gemini
import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim
from jittor.dataset import DataLoader

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model
from dataset.exporter import Exporter
from models.loss import L1Loss # Changed: Import specific loss

from tensorboardX import SummaryWriter

import time

jt.flags.use_cuda = 1

def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.lr_warmup_epochs:
        lr = args.learning_rate
    else:
        decay_progress = (epoch - args.lr_warmup_epochs) / (args.epochs - args.lr_warmup_epochs)
        lr_range = args.learning_rate - args.lr_end
        lr = args.learning_rate - lr_range * decay_progress
    optimizer.lr = lr
    return lr

def train(args):
    
    log_file = None
    if jt.rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        with open(log_file, 'w') as f:
            f.write(f"Starting MPI training with {jt.world_size} processes...\n")

    def log_message(message):
        if jt.rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
            print(message)

    log_message(f"Running with parameters: {args}")
    log_message(f"Global batch size: {args.batch_size} across {jt.world_size} GPUs.")

    model = create_model(model_name=args.model_name)

    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # --- [CHANGED] Use a mix of MSE and L1 loss ---
    criterion_mse = nn.MSELoss()
    criterion_l1 = L1Loss()

    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024+512, vertex_samples=512+216),
        transform=transform,
    )

    val_loader = None
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024+512, vertex_samples=512+216),
            transform=transform,
        )
        
    if jt.rank == 0:
        writer = SummaryWriter(logdir=f"runs/{args.model_name}_skin_mpi_{time.time()}")

    best_loss = 99999999
    for epoch in range(args.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        model.train()
        
        # --- [CHANGED] Add accumulator for MSE loss ---
        epoch_train_loss_mse = jt.Var(0.0)
        epoch_train_loss_l1 = jt.Var(0.0)

        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']
            
            outputs = model(vertices, joints)

            # --- [CHANGED] Calculate and combine both losses ---
            loss_mse = criterion_mse(outputs, skin)
            loss_l1 = criterion_l1(outputs, skin)
            # --- [CHANGED] Calculate and combine both losses --- Using simple L1 loss
            loss = 0 * loss_mse + loss_l1

            optimizer.step(loss)
            
            epoch_train_loss_mse += loss_mse
            epoch_train_loss_l1 += loss_l1

            if (batch_idx + 1) % args.print_freq == 0:
                    if jt.rank == 0:
                        # --- [CHANGED] Log both loss components ---
                        log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                                    f"Loss MSE: {loss_mse.item():.6f} L1: {loss_l1.item():.6f}")

        # --- Aggregate and Log Training Loss ---
        avg_train_loss_mse_local = epoch_train_loss_mse / len(train_loader)
        avg_train_loss_l1_local = epoch_train_loss_l1 / len(train_loader)

        if jt.in_mpi:
            final_avg_train_mse = avg_train_loss_mse_local.mpi_all_reduce(op='mean')
            final_avg_train_l1 = avg_train_loss_l1_local.mpi_all_reduce(op='mean')
        else:
            final_avg_train_mse = avg_train_loss_mse_local
            final_avg_train_l1 = avg_train_loss_l1_local

        if jt.rank == 0:
            epoch_time = time.time() - start_time
            # --- [CHANGED] Log both average losses for the epoch ---
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                        f"Train Loss MSE: {final_avg_train_mse.item():.6f} "
                        f"L1: {final_avg_train_l1.item():.6f} "
                        f"Time: {epoch_time:.2f}s "
                        f"LR: {current_lr:.6f}")

        # --- Validation Phase ---
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            # --- [CHANGED] Add accumulator for validation MSE loss ---
            epoch_val_loss_mse = jt.Var(0.0)
            epoch_val_loss_l1 = jt.Var(0.0)
            
            for val_batch_idx, val_data in enumerate(val_loader):
                vertices, joints, skin = val_data['vertices'], val_data['joints'], val_data['skin']
                outputs = model(vertices, joints)
                
                # --- [CHANGED] Calculate both losses for validation ---
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin) 
                
                epoch_val_loss_mse += loss_mse
                epoch_val_loss_l1 += loss_l1
                
                if jt.rank == 0 and args.export_render and val_batch_idx == 0:
                    exporter = Exporter()
                    render_path = f"tmp/skin_mpi/epoch_{epoch}"
                    os.makedirs(render_path, exist_ok=True)
                    for i in id_to_name:
                        name = id_to_name[i]
                        exporter._render_skin(path=f"{render_path}/{name}_ref.png", vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                        exporter._render_skin(path=f"{render_path}/{name}_pred.png", vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

            avg_val_loss_mse_local = epoch_val_loss_mse / len(val_loader)
            avg_val_loss_l1_local = epoch_val_loss_l1 / len(val_loader)

            if jt.in_mpi:
                final_avg_val_mse = avg_val_loss_mse_local.mpi_all_reduce(op='mean')
                final_avg_val_l1 = avg_val_loss_l1_local.mpi_all_reduce(op='mean')
            else:
                final_avg_val_mse = avg_val_loss_mse_local
                final_avg_val_l1 = avg_val_loss_l1_local
            
            if jt.rank == 0:
                avg_val_l1_value = final_avg_val_l1.item()
                # --- [CHANGED] Log both validation losses ---
                log_message(f"Validation Loss --> MSE: {final_avg_val_mse.item():.6f} L1: {avg_val_l1_value:.6f}")
                
                writer.add_scalar('Loss/train_mse', final_avg_train_mse.item(), epoch)
                writer.add_scalar('Loss/train_l1', final_avg_train_l1.item(), epoch)
                writer.add_scalar('Loss/val_mse', final_avg_val_mse.item(), epoch)
                writer.add_scalar('Loss/val_l1', avg_val_l1_value, epoch)
                
                # Best model is still saved based on L1 loss, as it's the key metric
                if avg_val_l1_value < best_loss:
                    best_loss = avg_val_l1_value
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with L1 loss {best_loss:.6f} to {model_path}")

        if jt.rank == 0 and (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    if jt.rank == 0:
        final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
        model.save(final_model_path)
        log_message(f"Training completed. Saved final model to {final_model_path}")
        
        writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train a point cloud model with jittor.mpi and learning rate scheduling')
    
    parser.add_argument('--train_data_list', type=str, required=True, help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='', help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for the data files')
    parser.add_argument('--model_name', type=str, default='ptcnn', choices=['ptcnn', 'pct', 'pct2', 'pct3','pctlast', 'ptcnn_adv'], help='Model architecture to use')
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=128, help='Global batch size for training across all GPUs')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate (at the beginning of training)')
    parser.add_argument('--lr_end', type=float, default=5e-5, help='Final learning rate (at the end of training)') # 调了一下学习率调度
    parser.add_argument('--lr_warmup_epochs', type=int, default=10, help='Number of epochs to keep the initial learning rate')
    parser.add_argument('--output_dir', type=str, default='output/skin_mpi_tensorboard', help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--export_render', action='store_true', default=True, help='Export render results during validation (can be slow).')
    
    args = parser.parse_args()
    
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed + jt.rank)
    np.random.seed(seed + jt.rank)
    random.seed(seed + jt.rank)

if __name__ == '__main__':
    seed_all(123)
    main()
