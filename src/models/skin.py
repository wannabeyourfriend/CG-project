import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from math import sqrt

# 导入PCT模型组件（假设已正确安装PCT库）
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group, Point_Transformer_v3, PTCNN
from PCT.misc.ops import knn_point, index_points, square_distance, topk


# 注意：Attention类已在skeleton_models.py中定义，此处通过导入使用
from .skeleton import Attention  # 假设已正确处理跨文件导入


class MLP(nn.Module):
    """多层感知机模块"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def execute(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(-1, self.input_dim)).reshape(B, -1, self.output_dim)


class PTCNNSkinModel_Advanced(nn.Module):
    """基于PTCNN的高级蒙皮模型"""
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.feat_dim = feat_dim  # PTCNN融合后的特征维度（1024）
        self.num_joints = num_joints
        
        # 复用PTCNN特征提取层
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False); self.bn2 = nn.BatchNorm1d(64)
        self.gather0 = Local_op(128, 128); self.sa0 = SA_Layer(128)
        self.gather1 = Local_op(256, 256); self.sa1 = SA_Layer(256)
        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, self.feat_dim, 1, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.LeakyReLU(scale=0.2),
        )
        self.relu = nn.ReLU()
        
        # 关节点特征生成模块
        self.joint_pos_embed = nn.Linear(3, self.feat_dim)  # 关节点坐标位置编码
        self.joint_attention = Attention(embed_dim=self.feat_dim, num_heads=8)  # 引用Attention类
        
        # 逐点特征生成模块
        self.vertex_feature_mlp = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(),
        )

    def execute(self, vertices: jt.Var, joints: jt.Var):
        B, N, _ = vertices.shape
        xyz = vertices  # (B, N, 3)
        
        # PTCNN特征提取
        x = vertices.transpose(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x))); x = self.relu(self.bn2(self.conv2(x)))
        feat = x.permute(0, 2, 1)
        new_xyz_1, new_feat_1 = sample_and_group(512, 32, xyz, feat)
        feat0 = self.sa0(self.gather0(new_feat_1), new_xyz_1.permute(0, 2, 1))
        new_xyz_2, new_feat_2 = sample_and_group(256, 32, new_xyz_1, feat0.permute(0, 2, 1))
        feat1 = self.sa1(self.gather1(new_feat_2), new_xyz_2.permute(0, 2, 1))
        pt_feat = self.pt_last(feat1, new_xyz_2)
        keypoint_features = self.conv_fuse(concat([pt_feat, feat1], dim=1))  # (B, feat_dim, 256)
        keypoint_xyz = new_xyz_2  # (B, 256, 3)
        
        # 特征传播：从关键点插值到原始点
        sqrdists = square_distance(vertices, keypoint_xyz)
        dist, idx = topk(sqrdists, 3, dim=-1, largest=False)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = jt.sum(dist_recip, dim=2, keepdims=True)
        weight = dist_recip / norm
        interpolated_features = jt.sum(
            index_points(keypoint_features.transpose(0, 2, 1), idx) * weight.view(B, N, 3, 1), 
            dim=2
        )
        vertex_features = self.vertex_feature_mlp(concat([vertices, interpolated_features], dim=-1))  # (B, N, feat_dim)
        
        # 关节点特征生成：交叉注意力
        joint_queries = self.joint_pos_embed(joints)  # (B, J, feat_dim)
        joint_features = self.joint_attention(joint_queries, keypoint_features.transpose(0, 2, 1), keypoint_features.transpose(0, 2, 1))  # (B, J, feat_dim)
        
        # 蒙皮权重计算：点积相似度+softmax
        attn_logits = vertex_features @ joint_features.transpose(0, 2, 1)  # (B, N, J)
        res = nn.softmax(attn_logits / sqrt(self.feat_dim), dim=-1)
        assert not jt.isnan(res).any()
        return res


class PTCNNSkinModel(nn.Module):
    """基于PTCNN的基础蒙皮模型"""
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.pct = PTCNN(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        # 提取全局形状特征
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))  # (B, feat_dim)
        
        # 顶点特征与关节点特征生成
        vertices_latent = self.vertex_mlp(concat([
            vertices, 
            shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)
        ], dim=-1))  # (B, N, feat_dim)
        joints_latent = self.joint_mlp(concat([
            joints, 
            shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)
        ], dim=-1))  # (B, J, feat_dim)
        
        # 计算蒙皮权重
        res = nn.softmax(
            vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), 
            dim=-1
        )
        assert not jt.isnan(res).any()
        return res


# 基础蒙皮模型变体
class SimpleSkinModel(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.pct = Point_Transformer(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))
        vertices_latent = self.vertex_mlp(concat([
            vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)
        ], dim=-1))
        joints_latent = self.joint_mlp(concat([
            joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)
        ], dim=-1))
        res = nn.softmax(
            vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), 
            dim=-1
        )
        assert not jt.isnan(res).any()
        return res


class SimpleSkinModel2(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.pct = Point_Transformer2(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))
        vertices_latent = self.vertex_mlp(concat([
            vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)
        ], dim=-1))
        joints_latent = self.joint_mlp(concat([
            joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)
        ], dim=-1))
        res = nn.softmax(
            vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), 
            dim=-1
        )
        assert not jt.isnan(res).any()
        return res


class SimpleSkinModel3(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.pct = Point_Transformer_v3(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))
        vertices_latent = self.vertex_mlp(concat([
            vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)
        ], dim=-1))
        joints_latent = self.joint_mlp(concat([
            joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)
        ], dim=-1))
        res = nn.softmax(
            vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), 
            dim=-1
        )
        assert not jt.isnan(res).any()
        return res


class SimpleSkinModellast(nn.Module):
    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.pct = Point_Transformer_Last(output_channels=feat_dim)
        self.joint_mlp = MLP(3 + feat_dim, feat_dim)
        self.vertex_mlp = MLP(3 + feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def execute(self, vertices: jt.Var, joints: jt.Var):
        shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))
        vertices_latent = self.vertex_mlp(concat([
            vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)
        ], dim=-1))
        joints_latent = self.joint_mlp(concat([
            joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)
        ], dim=-1))
        res = nn.softmax(
            vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), 
            dim=-1
        )
        assert not jt.isnan(res).any()
        return res


# 蒙皮模型工厂函数
def create_model(model_name='ptcnn', feat_dim=256, **kwargs):
    if model_name == "ptcnn":
        return PTCNNSkinModel(feat_dim=feat_dim, num_joints=22)
    if model_name == "pct":
        return SimpleSkinModel(feat_dim=feat_dim, num_joints=22)
    if model_name == "pct2":
        return SimpleSkinModel2(feat_dim=feat_dim, num_joints=22)
    if model_name == "pct3":
        return SimpleSkinModel3(feat_dim=feat_dim, num_joints=22)
    if model_name == "pctlast":
        return SimpleSkinModellast(feat_dim=feat_dim, num_joints=22)
    if model_name == "ptcnn_adv":
        return PTCNNSkinModel_Advanced(feat_dim=1024, num_joints=22)
    raise NotImplementedError(f"未实现的模型: {model_name}")