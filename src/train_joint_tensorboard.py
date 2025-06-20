import jittor as jt
from jittor import nn, optim
import argparse
import os
from datetime import datetime
import numpy as np
from jittor.dataset import Dataset
import copy # 导入 copy 模块

# ======================================================================================
# 依赖项导入
# ======================================================================================
from typing import Tuple, Dict
from abc import ABC, abstractmethod

# (Sampler and sample_surface functions remain the same as before)
class Sampler(ABC):
    """Abstract base class for samplers."""
    def __init__(self):
        pass

    def _sample_barycentric(self, vertex_groups: np.ndarray, faces: np.ndarray, face_index: np.ndarray, random_lengths: np.ndarray):
        """Helper to sample vertex attributes using barycentric coordinates."""
        v_origins = vertex_groups[faces[face_index, 0]]
        v_vectors = vertex_groups[faces[face_index, 1:]]
        v_vectors -= v_origins[:, np.newaxis, :]
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples

    @abstractmethod
    def sample(self, vertices: np.ndarray, vertex_normals: np.ndarray, face_normals: np.ndarray, vertex_groups: Dict[str, np.ndarray], faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Abstract sample method."""
        return vertices, vertex_normals, vertex_groups

def sample_surface(num_samples: int, vertices: np.ndarray, faces: np.ndarray, return_weight: bool=False):
    """
    Randomly sample points on a mesh surface, weighted by face area.
    """
    if num_samples <= 0:
        empty_f3 = np.zeros((0, 3), dtype=np.float32)
        empty_f21 = np.zeros((0, 2, 1), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        if not return_weight: return empty_f3
        return empty_f3, empty_i, empty_f21
    
    vec_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    vec_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_areas = np.linalg.norm(np.cross(vec_a, vec_b), axis=1) / 2.0
    
    total_area = face_areas.sum()
    if total_area < 1e-9: # Handle case with zero area
        face_index = np.random.choice(len(faces), size=num_samples)
    else:
        prob = face_areas / total_area
        face_index = np.random.choice(len(faces), size=num_samples, p=prob)
    
    triangles = vertices[faces[face_index]]
    
    u = np.random.rand(num_samples, 1)
    v = np.random.rand(num_samples, 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    
    points = u * triangles[:, 0, :] + v * triangles[:, 1, :] + w * triangles[:, 2, :]
    
    if not return_weight:
        return points
    
    random_lengths = np.hstack([u,v]).reshape(num_samples, 2, 1)
    return points, face_index, random_lengths


class SamplerMix(Sampler):
    """
    A sampler that combines sampling vertices directly and sampling from the mesh surface.
    """
    def __init__(self, num_samples: int, vertex_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples

    def __iter__(self):
        if not hasattr(self, 'dataset'): raise RuntimeError("Sampler not attached to a dataset.")
        return iter(np.random.permutation(len(self.dataset)))

    def __len__(self):
        return len(self.dataset) if hasattr(self, 'dataset') else 0

    def sample(self, vertices: np.ndarray, vertex_normals: np.ndarray, face_normals: np.ndarray, vertex_groups: Dict[str, np.ndarray], faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        if self.num_samples == -1: return vertices, vertex_normals, vertex_groups

        num_surface_samples = self.num_samples
        perm = np.random.permutation(vertices.shape[0])
        num_direct_samples = min(self.vertex_samples, vertices.shape[0])
        num_surface_samples -= num_direct_samples
        
        direct_indices = perm[:num_direct_samples]
        direct_vertices = vertices[direct_indices]
        direct_normals = vertex_normals[direct_indices]
        direct_groups = {name: v[direct_indices] for name, v in vertex_groups.items()}

        surface_vertices, face_indices, random_lengths = sample_surface(
            num_samples=num_surface_samples, vertices=vertices, faces=faces, return_weight=True
        )
        surface_normals = face_normals[face_indices]
        surface_groups = {}
        for name, v_group in vertex_groups.items():
            g = self._sample_barycentric(vertex_groups=v_group, faces=faces, face_index=face_indices, random_lengths=random_lengths)
            surface_groups[name] = g

        all_vertices = np.concatenate([direct_vertices, surface_vertices], axis=0)
        all_normals = np.concatenate([direct_normals, surface_normals], axis=0)
        all_groups = {name: np.concatenate([direct_groups[name], surface_groups[name]], axis=0) for name in vertex_groups}

        return all_vertices, all_normals, all_groups

# 从模型文件中导入 LBS 和模型
from models.joint import AdvancedJointModel, DifferentiableLBS
from tensorboardX import SummaryWriter

jt.flags.use_cuda = 1

# ======================================================================================
# Asset 类
# ======================================================================================
class Asset:
    """包含关节体数据的容器，包括骨骼层级结构。"""
    def __init__(self, data_dict):
        self.vertices = data_dict.get('vertices')
        self.faces = data_dict.get('faces')
        self.skin = data_dict.get('skin')
        self.joints = data_dict.get('joints')
        self.parent = data_dict.get('parent')
        self.matrix_local = data_dict.get('matrix_local')
        self.matrix_world = data_dict.get('matrix_world')
        self.vertex_normals = data_dict.get('vertex_normals', np.zeros_like(self.vertices) if self.vertices is not None else None)
        self.face_normals = data_dict.get('face_normals', np.zeros_like(self.faces) if self.faces is not None else None)

    @classmethod
    def load(cls, path):
        """从单个 .npz 文件加载 asset。"""
        try:
            data = np.load(path, allow_pickle=True)
            data_dict = {key: data[key] for key in data.files}
            asset_instance = cls(data_dict)
            return asset_instance
        except Exception as e:
            print(f"Error loading or processing asset file {path}: {e}")
            return None
    
    def complete_skeleton_info(self, parent_hierarchy):
        """使用固定的父节点层级来计算 T-pose 的局部和世界变换矩阵。"""
        self.parent = parent_hierarchy
        self.matrix_local = np.tile(np.eye(4), (self.joints.shape[0], 1, 1))
        
        root_indices = np.where(self.parent == -1)[0]
        for root_idx in root_indices:
            self.matrix_local[root_idx, :3, 3] = self.joints[root_idx]
        
        for i in range(self.joints.shape[0]):
            if self.parent[i] != -1:
                # Ensure parent exists to prevent IndexError
                if self.parent[i] < len(self.joints):
                    offset = self.joints[i] - self.joints[self.parent[i]]
                    self.matrix_local[i, :3, 3] = offset
                else:
                    # Handle invalid parent index if necessary
                    self.matrix_world = None
                    return
        
        self.recompute_matrix_world()

    def recompute_matrix_world(self):
        """根据局部矩阵和层级关系，按正确的拓扑顺序重新计算世界变换矩阵。"""
        if self.matrix_local is None or self.parent is None:
            self.matrix_world = None
            return
        
        self.matrix_world = np.zeros_like(self.matrix_local)
        
        children = {i: [] for i in range(len(self.parent))}
        roots = []
        for i, p in enumerate(self.parent):
            if p == -1:
                roots.append(i)
            else:
                if p < len(children):
                    children[p].append(i)
                else: # Invalid parent index
                    self.matrix_world = None
                    return
        
        queue = roots[:]
        
        for root_idx in roots:
            self.matrix_world[root_idx] = self.matrix_local[root_idx]
            
        head = 0
        while head < len(queue):
            p_idx = queue[head]
            head += 1
            for c_idx in children[p_idx]:
                self.matrix_world[c_idx] = self.matrix_world[p_idx] @ self.matrix_local[c_idx]
                queue.append(c_idx)

    def copy(self):
        """返回此对象的深拷贝。"""
        return copy.deepcopy(self)

def linear_blend_skinning(V, W, J_inv_mats, pose):
    """在CPU上执行线性混合蒙皮。"""
    T = pose @ J_inv_mats
    V_hom = np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
    V_posed_hom = np.einsum('nj,jxy,ny->nx', W, T, V_hom)
    V_posed = V_posed_hom[:, :3] / (V_posed_hom[:, 3, np.newaxis] + 1e-8)
    return V_posed

# ======================================================================================
# 数据集实现
# ======================================================================================
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])

def transform_to_unit_cube(asset: Asset):
    """将 asset 的顶点和关节归一化到 [-1, 1]^3 的立方体中。"""
    if asset.vertices is None or asset.vertices.shape[0] == 0:
        return asset, np.zeros(3), 1.0
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    center = (min_vals + max_vals) / 2
    scale = np.max(max_vals - min_vals) / 2
    if scale < 1e-6: scale = 1.0
    
    asset.vertices = (asset.vertices - center) / scale
    if asset.joints is not None:
        asset.joints = (asset.joints - center) / scale

    if asset.matrix_world is not None:
        asset.matrix_world[:, :3, 3] = (asset.matrix_world[:, :3, 3] - center) / scale
    if asset.matrix_local is not None:
        root_indices = np.where(asset.parent == -1)[0]
        for root_idx in root_indices:
                asset.matrix_local[root_idx, :3, 3] = (asset.matrix_local[root_idx, :3, 3] - center) / scale
    return asset, center, scale

class ArticulatedBodyDataset(Dataset):
    """为基于LBS的训练准备数据的数据集。"""
    def __init__(self, data_root, mode, n_points, n_joints):
        super().__init__()
        self.data_root = data_root
        self.n_points = n_points
        self.n_joints = n_joints
        
        list_file = os.path.join(data_root, f"{mode}_list.txt")
        if not os.path.exists(list_file): raise FileNotFoundError(f"Data list file not found: {list_file}")
        with open(list_file, 'r') as f: self.paths = [line.strip() for line in f.readlines()]
            
        self.sampler = SamplerMix(num_samples=self.n_points, vertex_samples=self.n_points // 4)
        self.transform = transform_to_unit_cube
        self.set_attrs(total_len=len(self.paths), shuffle=True, drop_last=True)

    def __getitem__(self, index):
        path = os.path.join(self.data_root, self.paths[index])
        asset = Asset.load(path)

        if asset is None or asset.joints is None or asset.joints.shape[0] != self.n_joints:
            print(f"Skipping corrupted or invalid asset: {path}")
            return self.__getitem__((index + 1) % len(self.paths))

        asset.complete_skeleton_info(SMPL_PARENTS)

        t_pose_asset, _center, _scale = self.transform(asset.copy())
        
        if t_pose_asset.matrix_world is None:
            print(f"Skipping asset due to matrix generation failure: {path}")
            return self.__getitem__((index + 1) % len(self.paths))

        gt_joints = jt.array(t_pose_asset.joints.copy()).float32()
        
        # 1. 采样点和对应的权重
        sampled_points_np, _normals, sampled_groups = self.sampler.sample(
            vertices=t_pose_asset.vertices,
            vertex_normals=t_pose_asset.vertex_normals,
            face_normals=t_pose_asset.face_normals,
            vertex_groups={'skin': t_pose_asset.skin},
            faces=t_pose_asset.faces
        )
        points = jt.array(sampled_points_np).float32()
        gt_weights = jt.array(sampled_groups['skin']).float32()

        # 2. 生成随机姿态
        posed_asset = t_pose_asset.copy()
        num_joints_to_rotate = min(3, self.n_joints)
        joint_indices = np.random.choice(self.n_joints, num_joints_to_rotate, replace=False)
        
        new_local_matrices = posed_asset.matrix_local.copy()
        for i in joint_indices:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            axis_angle = np.random.randn(3)
            axis_angle /= (np.linalg.norm(axis_angle) + 1e-8)
            
            from scipy.spatial.transform import Rotation as R
            rot = R.from_rotvec(axis_angle * angle).as_matrix()
            
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot
            new_local_matrices[i] = new_local_matrices[i] @ rot_4x4
        
        posed_asset.matrix_local = new_local_matrices
        posed_asset.recompute_matrix_world()
        
        gt_pose = jt.array(posed_asset.matrix_world).float32()
        gt_J_inv_mats = jt.array(np.linalg.inv(t_pose_asset.matrix_world)).float32()

        # 3. 计算姿态变换后的 *采样点* 作为真值 (Ground Truth)
        gt_posed_sampled_points_np = linear_blend_skinning(
            V=sampled_points_np, W=sampled_groups['skin'],
            J_inv_mats=np.linalg.inv(t_pose_asset.matrix_world),
            pose=posed_asset.matrix_world
        )
        gt_posed_points = jt.array(gt_posed_sampled_points_np).float32()

        return points, gt_joints, gt_weights, gt_posed_points, gt_J_inv_mats, gt_pose

# ======================================================================================
# 损失函数, 训练流程, 主函数
# ======================================================================================

def J2J(joints_a: jt.Var, joints_b: jt.Var) -> jt.Var:
    '''
    Calculate batched J2J loss in [-1, 1]^3 cube.
    '''
    assert isinstance(joints_a, jt.Var) and isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 3 and joints_b.ndim == 3, "Input tensors should be batched (B, J, 3)"
    
    joints_a_exp = joints_a.unsqueeze(2)
    joints_b_exp = joints_b.unsqueeze(1)
    
    dist_matrix = ((joints_a_exp - joints_b_exp)**2).sum(dim=-1).sqrt()
    
    min_dist_a = dist_matrix.min(dim=2)
    loss1 = min_dist_a.mean(dim=1)
    
    min_dist_b = dist_matrix.min(dim=1)
    loss2 = min_dist_b.mean(dim=1)

    batch_loss = (loss1 + loss2) / 4.0
    return batch_loss.mean()

def train_step(model, lbs_layer, optimizer, data, l1_loss_fn, loss_weights):
    model.train()
    points, gt_joints, gt_weights, gt_posed_points, gt_J_inv_mats, gt_pose = data
    
    pred_joints, pred_weights = model(points)
    
    # 使用真实的骨骼位姿来计算顶点损失，以稳定训练
    pred_posed_points = lbs_layer(points, pred_weights, gt_J_inv_mats, gt_pose)
    
    w_j, w_s, w_v = loss_weights
    
    loss_j = J2J(pred_joints, gt_joints)
    loss_s = l1_loss_fn(pred_weights, gt_weights)
    loss_v = l1_loss_fn(pred_posed_points, gt_posed_points)
    
    total_loss = w_j * loss_j + w_s * loss_s + w_v * loss_v
    optimizer.backward(total_loss)
    optimizer.step()
    return {'total_loss': total_loss.item(), 'j2j_loss': loss_j.item(), 'skin_l1_loss': loss_s.item(), 'vertex_l1_loss': loss_v.item()}

# 新增：辅助函数，用于从预测的骨骼关节点计算逆绑定矩阵
def get_inv_bind_pose_from_joints(joints_batch_numpy, parents):
    """
    Computes inverse bind pose matrices from joint positions for a batch.
    This is a non-differentiable, CPU-based operation for validation.
    """
    batch_size = joints_batch_numpy.shape[0]
    num_joints = joints_batch_numpy.shape[1]
    inv_bind_poses = np.zeros((batch_size, num_joints, 4, 4), dtype=np.float32)

    for i in range(batch_size):
        joints = joints_batch_numpy[i]
        # 使用临时的 Asset 对象来运行前向动力学逻辑
        temp_asset = Asset({'joints': joints})
        temp_asset.complete_skeleton_info(parents)
        
        if temp_asset.matrix_world is not None:
            try:
                inv_bind_poses[i] = np.linalg.inv(temp_asset.matrix_world)
            except np.linalg.LinAlgError:
                # 如果矩阵是奇异的，则回退到单位矩阵
                print(f"Warning: Singular matrix encountered in validation for batch item {i}. Using identity.")
                inv_bind_poses[i] = np.tile(np.eye(4), (num_joints, 1, 1))
        else:
            # 如果矩阵计算失败，则回退
            print(f"Warning: World matrix computation failed in validation for batch item {i}. Using identity.")
            inv_bind_poses[i] = np.tile(np.eye(4), (num_joints, 1, 1))
            
    return inv_bind_poses

# 已更新：validate_step 现在计算两个版本的顶点损失
def validate_step(model, lbs_layer, data, l1_loss_fn):
    model.eval()
    points, gt_joints, gt_weights, gt_posed_points, gt_J_inv_mats, gt_pose = data
    with jt.no_grad():
        pred_joints, pred_weights = model(points)
        
        # --- 损失 1：解耦损失 (用于诊断) ---
        # 这是原始的计算方式，使用真实的骨骼变换矩阵
        # 它可以纯粹地衡量权重预测的质量，不受骨骼预测误差的影响
        pred_posed_points_decoupled = lbs_layer(points, pred_weights, gt_J_inv_mats, gt_pose)
        loss_j = J2J(pred_joints, gt_joints)
        loss_s = l1_loss_fn(pred_weights, gt_weights)
        loss_v_decoupled = l1_loss_fn(pred_posed_points_decoupled, gt_posed_points)
        
        # --- 损失 2：耦合损失 (用于最终评分) ---
        # 这里的计算同时使用了预测的骨骼和预测的权重，真实地反映了模型的端到端性能
        # 1. 从预测的骨骼关节点(pred_joints)计算逆绑定矩阵
        pred_joints_np = pred_joints.numpy()
        pred_J_inv_mats_np = get_inv_bind_pose_from_joints(pred_joints_np, SMPL_PARENTS)
        pred_J_inv_mats = jt.array(pred_J_inv_mats_np)
        
        # 2. 使用预测的骨骼矩阵和预测的权重计算顶点位置
        pred_posed_points_coupled = lbs_layer(points, pred_weights, pred_J_inv_mats, gt_pose)
        
        # 3. 计算用于评分的耦合顶点损失
        loss_v_coupled = l1_loss_fn(pred_posed_points_coupled, gt_posed_points)

    return {
        'val_j2j_loss': loss_j.item(), 
        'val_skin_l1_loss': loss_s.item(), 
        'val_vertex_l1_loss': loss_v_decoupled.item(),       # 解耦损失 (用于诊断)
        'val_vertex_l1_loss_score': loss_v_coupled.item()    # 耦合损失 (用于评分)
    }

def calculate_score(cd_j2j_loss, skin_l1_loss, vertex_l1_loss):
    cd_j2j_metric = cd_j2j_loss
    term1 = 1.0 / min(cd_j2j_metric, 0.01) if cd_j2j_metric > 1e-6 else 100.0
    term2 = 0.5 * (1.0 - 20.0 * min(skin_l1_loss, 0.05))
    ver_loss_clamped = min(vertex_l1_loss, 1.0)
    term3 = 0.5 * (1.0 - np.sqrt(ver_loss_clamped))
    score = term1 * (term2 + term3)
    return score

def get_loss_weights(epoch, args):
    """
    根据当前 epoch 动态获取损失权重。
    - 前 5 个 epoch: 只优化骨骼预测 (J2J loss)。
    - 10 到 20 epoch: 只优化蒙皮权重。
    - 20 epoch 之后: 多目标优化。
    """
    if epoch < 3:
        # 阶段 1: 只关注骨骼预测
        return 1.0, 0.0, 0.0
    elif epoch < 10:
        # 阶段 2: 只关注蒙皮权重
        return 0.0, args.w_s, 0.0
    else:
        # 阶段 3: 多目标联合优化
        return args.w_j, args.w_s, args.w_v * 0

def main():
    parser = argparse.ArgumentParser(description="Train a joint skeleton and skinning model with LBS.")
    parser.add_argument('--data_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('--save_dir', type=str, default='./output/joint_tensorboard', help='path to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs', help='path to save tensorboard logs')
    parser.add_argument('--exp_name', type=str, default=datetime.now().strftime('%Y%m%d-%H%M%S'), help='experiment name')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points per shape')
    parser.add_argument('--num_joints', type=int, default=22, help='number of joints')
    parser.add_argument('--feat_dim', type=int, default=1024, help='feature dimension for the backbone')
    parser.add_argument('--w_j', type=float, default=1.0, help='weight for joint chamfer loss')
    parser.add_argument('--w_s', type=float, default=1.0, help='weight for skinning L1 loss')
    parser.add_argument('--w_v', type=float, default=1.0, help='weight for vertex L1 loss (LBS)')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))
    print(f"TensorBoard logs will be saved to: {os.path.join(args.log_dir, args.exp_name)}")

    train_dataset = ArticulatedBodyDataset(args.data_dir, mode='train', n_points=args.num_points, n_joints=args.num_joints)
    train_loader = train_dataset.set_attrs(batch_size=args.batch_size, shuffle=True)
    val_dataset = ArticulatedBodyDataset(args.data_dir, mode='val', n_points=args.num_points, n_joints=args.num_joints)
    val_loader = val_dataset.set_attrs(batch_size=args.batch_size, shuffle=False)
    
    model = AdvancedJointModel(num_joints=args.num_joints, feat_dim=args.feat_dim)
    lbs_layer = DifferentiableLBS()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    l1_loss_fn = nn.L1Loss()
    
    start_epoch, global_step = 0, 0
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = jt.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', start_epoch * len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        # *** 新增: 在每个 epoch 开始时获取动态权重 ***
        loss_weights = get_loss_weights(epoch, args)
        w_j, w_s, w_v = loss_weights
        print(f"Epoch {epoch+1}/{args.epochs} - Using loss weights: J2J={w_j}, Skin={w_s}, Vertex={w_v}")

        train_loss_dict = {k: [] for k in ['total_loss', 'j2j_loss', 'skin_l1_loss', 'vertex_l1_loss']}
        
        for i, data in enumerate(train_loader):
            step_losses = train_step(model, lbs_layer, optimizer, data, l1_loss_fn, loss_weights)
            for k,v in step_losses.items(): 
                train_loss_dict[k].append(v)
                if i % 10 == 0: writer.add_scalar(f'Loss/train_{k}', v, global_step)
            if i % 20 == 0: print(f"Epoch {epoch+1}/{args.epochs}, Step {i}/{len(train_loader)}, Loss: {step_losses['total_loss']:.4f}")
            global_step += 1
        
        # 已更新：初始化字典以包含新的评分损失
        val_loss_dict = {k: [] for k in ['val_j2j_loss', 'val_skin_l1_loss', 'val_vertex_l1_loss', 'val_vertex_l1_loss_score']}
        for data in val_loader:
            step_losses = validate_step(model, lbs_layer, data, l1_loss_fn)
            for k, v in step_losses.items(): val_loss_dict[k].append(v)
        
        avg_val_j2j = np.mean(val_loss_dict['val_j2j_loss'])
        avg_val_skin = np.mean(val_loss_dict['val_skin_l1_loss'])
        avg_val_vertex_decoupled = np.mean(val_loss_dict['val_vertex_l1_loss'])
        avg_val_vertex_score = np.mean(val_loss_dict['val_vertex_l1_loss_score'])
        
        # 已更新：使用耦合损失 (avg_val_vertex_score) 来计算分数
        score = calculate_score(avg_val_j2j, avg_val_skin, avg_val_vertex_score)
        
        writer.add_scalar('Score/validation', score, epoch)
        writer.add_scalar('ValLoss/j2j', avg_val_j2j, epoch)
        writer.add_scalar('ValLoss/skin_l1', avg_val_skin, epoch)
        # 已更新：同时记录两个版本的顶点损失以供分析
        writer.add_scalar('ValLoss/vertex_l1_decoupled', avg_val_vertex_decoupled, epoch)
        writer.add_scalar('ValLoss/vertex_l1_for_score', avg_val_vertex_score, epoch)
        
        print(f"Epoch {epoch+1} Summary: Score: {score:.2f}, Val Losses (J/S/V_score): {avg_val_j2j:.4f}/{avg_val_skin:.4f}/{avg_val_vertex_score:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pkl')
            print(f"Saving model to {save_path}")
            jt.save({
                'epoch': epoch + 1, 'global_step': global_step, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': args
            }, save_path)
    
    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()
