# 2025/6/18 by Wang Zixuan
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
from dataset.sampler import SamplerMix
# from dataset.sampler2 import SamplerMix2 # 新的点云采样，采用FPS技术，帮助模型更好地捕捉mesh表面
from dataset.exporter import Exporter
from models.skeleton import create_model
from models.metrics import J2J
from models.loss import * # 引入更多loss

from tensorboardX import SummaryWriter

import time

jt.flags.use_cuda = 1

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate for the optimizer based on the current epoch."""
    if epoch < args.lr_warmup_epochs:
        lr = args.learning_rate
    else:
        decay_progress = (epoch - args.lr_warmup_epochs) / (args.epochs - args.lr_warmup_epochs)
        lr_range = args.learning_rate - args.lr_end
        lr = args.learning_rate - lr_range * decay_progress
    optimizer.lr = lr
    return lr

def train(args):
    """Main training function adapted for MPI."""
    
    log_file = None
    if jt.rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        with open(log_file, 'w') as f:
            f.write(f"Starting MPI training with {jt.world_size} processes...\n")

    def log_message(message):
        """Logs a message from the main process."""
        if jt.rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
            print(message)

    log_message(f"Running with parameters: {args}")
    log_message(f"Global batch size: {args.batch_size} across {jt.world_size} GPUs.")

    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    # sampler = SamplerMix2(num_samples=1536, vertex_samples=768)

    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # 2025/6/18 by Wang Zixuan
    # TODO: 修改loss？
    criterion = nn.MSELoss()
    # --- [CHANGED] Use ChamferDistanceLoss, which aligns with the J2J metric ---
    criterion2 = ChamferDistanceLoss()
    sym_loss_fn = SymmetricJointLoss(mode=args.symmetry_type, joint_pairs=[(6,10), (7,11), (8,12), (9,13),(14, 18), (15, 19), (16, 20), (17, 21)])

    
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
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
            sampler=sampler,
            transform=transform,
        )
        
    if jt.rank == 0:
        writer = SummaryWriter(logdir=f"runs/{args.model_name}_skeleton_mpi_{time.time()}")
    
    best_loss = 99999999
    for epoch in range(args.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        model.train()
        
        epoch_train_loss = jt.Var(0.0)
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices, joints = data['vertices'], data['joints']
            vertices = vertices.permute(0, 2, 1)  # Reshape to [B, 3, N]

            outputs = model(vertices)
            joints_flat = joints.reshape(outputs.shape[0], -1)

            loss_main = criterion(outputs, joints_flat)
            sym_loss = sym_loss_fn(outputs)
            loss = loss_main + args.symmetry_weight * sym_loss

            
            
            # TODO:
            # pred_joints = outputs.reshape(outputs.shape[0], -1, 3)
            # loss = criterion2(pred_joints, joints)
            
            
            optimizer.step(loss)
            
            epoch_train_loss += loss

            if (batch_idx + 1) % args.print_freq == 0:
                 if jt.rank == 0: # Print from main process only
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss_local = epoch_train_loss / len(train_loader)

        if jt.in_mpi:
            final_avg_train_loss = avg_train_loss_local.mpi_all_reduce(op='mean')
        else:
            final_avg_train_loss = avg_train_loss_local

        if jt.rank == 0:
            epoch_time = time.time() - start_time
            log_message(f"Epoch [{epoch+1}/{args.epochs}]" 
                        f"Train Loss: {final_avg_train_loss.item():.4f}"
                        f"Time: {epoch_time:.2f}s"
                        f"LR: {current_lr:.6f}")
            if args.symmetry_type != 'none':
                log_message(f"(Symmetry loss type: {args.symmetry_type}, weight: {args.symmetry_weight})")


        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            epoch_val_loss = jt.Var(0.0)
            epoch_j2j_loss = jt.Var(0.0)
            
            show_id = -1
            if jt.rank == 0 and len(val_loader) > 0:
                show_id = np.random.randint(0, len(val_loader))
            
            for val_batch_idx, val_data in enumerate(val_loader):
                vertices, joints = val_data['vertices'], val_data['joints']
                joints_flat = joints.reshape(joints.shape[0], -1)
                vertices_permuted = vertices.permute(0, 2, 1)

                outputs = model(vertices_permuted)
                loss = criterion(outputs, joints_flat)

                loss_main = criterion(outputs, joints_flat)
                sym_loss = sym_loss_fn(outputs)
                loss = loss_main + args.symmetry_weight * sym_loss

                epoch_val_loss += loss
                
                batch_j2j_loss = 0
                for i in range(outputs.shape[0]):
                    batch_j2j_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3))
                epoch_j2j_loss += batch_j2j_loss / outputs.shape[0]

                if jt.rank == 0 and args.export_render and val_batch_idx == show_id:
                    log_message(f"Rendering visualization for epoch {epoch+1}, batch {val_batch_idx}...")
                    exporter = Exporter()
                    render_path = f"tmp/skeleton_mpi/epoch_{epoch+1}"
                    os.makedirs(render_path, exist_ok=True)
                    from dataset.format import parents

                    vis_vertices = vertices[0].numpy()
                    vis_joints_ref = joints[0].numpy().reshape(-1, 3)
                    vis_joints_pred = outputs[0].numpy().reshape(-1, 3)
                    
                    exporter._render_skeleton(path=f"{render_path}/skeleton_ref.png", joints=vis_joints_ref, parents=parents)
                    exporter._render_skeleton(path=f"{render_path}/skeleton_pred.png", joints=vis_joints_pred, parents=parents)
                    exporter._render_pc(path=f"{render_path}/vertices.png", vertices=vis_vertices)


            avg_val_loss_local = epoch_val_loss / len(val_loader)
            avg_j2j_loss_local = epoch_j2j_loss / len(val_loader)

            if jt.in_mpi:
                final_avg_val_loss = avg_val_loss_local.mpi_all_reduce(op='mean')
                final_avg_j2j_loss = avg_j2j_loss_local.mpi_all_reduce(op='mean')
            else:
                final_avg_val_loss = avg_val_loss_local
                final_avg_j2j_loss = avg_j2j_loss_local

            if jt.rank == 0:
                avg_j2j_value = final_avg_j2j_loss.item()
                log_message(f"Validation Loss: {final_avg_val_loss.item():.4f} J2J Loss: {avg_j2j_value:.4f}")
                
                writer.add_scalar('Loss/Train', final_avg_train_loss.item(), epoch)
                writer.add_scalar('Loss/Val', final_avg_val_loss.item(), epoch)
                writer.add_scalar('Loss/J2J', avg_j2j_value, epoch)
                
                if avg_j2j_value < best_loss:
                    best_loss = avg_j2j_value
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with J2J loss {best_loss:.4f} to {model_path}")
        
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
    parser = argparse.ArgumentParser(description='Train a skeleton model with Jittor MPI')
    parser.add_argument('--train_data_list', type=str, required=True, help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='', help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for the data files')
    parser.add_argument('--model_name', type=str, default='ptcnn', choices=[ 'ptcnn', 'pct', 'pct2', 'pct3', 'pctlast', 'adv', 'ptcnn_adv'], help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'enhanced'], help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=256, help='Global batch size for training across all GPUs')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--symmetry_type', type=str, default='none',
                    choices=['none', 'position', 'structure'],
                    help='Symmetry constraint type (default: none)')
    parser.add_argument('--symmetry_weight', type=float, default=0.0,
                    help='Weight for symmetry loss (default: 0.0)')

    parser.add_argument('--learning_rate', type=float, default=2*1e-4, help='Initial learning rate')
    parser.add_argument('--lr_end', type=float, default=5*1e-5, help='Final learning rate')
    parser.add_argument('--lr_warmup_epochs', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--output_dir', type=str, default='output/skeleton_mpi_tensorboard', help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--export_render', action='store_true', default=1, help='Export render results during validation (can be slow).')
    
    args = parser.parse_args()
    
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed + jt.rank)
    np.random.seed(seed + jt.rank)
    random.seed(seed + jt.rank)

if __name__ == '__main__':
    seed_all(123)
    main()
