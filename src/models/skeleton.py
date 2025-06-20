import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from math import sqrt

# 导入PCT模型组件（假设已正确安装PCT库）
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group, Point_Transformer_v3, PTCNN
from PCT.misc.ops import knn_point, index_points, square_distance, topk


class Attention(nn.Module):
    """标准多头交叉注意力模块"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim必须能被num_heads整除"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, query, key, value):
        B, N_q, C = query.shape; B, N_k, C_k = key.shape
        # 线性投影与维度重组
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # 计算注意力得分
        attn_scores = jt.matmul(q, k) * (self.head_dim ** -0.5)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)
        # 加权聚合特征
        context = jt.matmul(attn_probs, v).permute(0, 2, 1, 3).reshape(B, N_q, C)
        return self.out_proj(context)


class AdvancedSkeletonModel(nn.Module):
    """带注意力机制的高级骨架模型"""
    def __init__(self, feat_dim: int, output_channels: int, num_joints: int = 22):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 骨干网络特征提取层
        self.backbone_conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.backbone_bn1 = nn.BatchNorm1d(128)
        self.backbone_conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.backbone_bn2 = nn.BatchNorm1d(128)
        self.backbone_sa1 = SA_Layer(128)
        self.backbone_sa2 = SA_Layer(128)
        self.backbone_sa3 = SA_Layer(128)
        self.backbone_sa4 = SA_Layer(128)
        self.backbone_conv_fuse = nn.Sequential(
            nn.Conv1d(512, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(scale=0.2)
        )
        self.relu = nn.ReLU()
        
        # 可学习的关节点查询向量
        self.joint_queries = nn.Parameter(jt.zeros((1, self.num_joints, self.feat_dim)))
        init.gauss_(self.joint_queries, 0, 1)
        
        # 交叉注意力层
        self.attention = Attention(embed_dim=feat_dim, num_heads=8)
        
        # 坐标回归头
        self.mlp_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 3)
        )

    def execute(self, vertices: jt.Var):
        # 输入维度调整：(B, N, 3) -> (B, 3, N)
        vertices_t = vertices.transpose(0, 2, 1)
        B, C, N = vertices_t.shape
        
        # 特征提取
        x = self.relu(self.backbone_bn1(self.backbone_conv1(vertices_t)))
        x = self.relu(self.backbone_bn2(self.backbone_conv2(x)))
        x1 = self.backbone_sa1(x, vertices_t)
        x2 = self.backbone_sa2(x1, vertices_t)
        x3 = self.backbone_sa3(x2, vertices_t)
        x4 = self.backbone_sa4(x3, vertices_t)
        x_cat = concat((x1, x2, x3, x4), dim=1)
        point_features = self.backbone_conv_fuse(x_cat)  # (B, feat_dim, N)
        
        # 交叉注意力：关节点查询点云特征
        point_features_t = point_features.transpose(0, 2, 1)  # (B, N, feat_dim)
        queries = self.joint_queries.repeat(B, 1, 1)  # (B, num_joints, feat_dim)
        joint_features = self.attention(queries, point_features_t, point_features_t)  # (B, num_joints, feat_dim)
        
        # 坐标回归
        pred_joints = self.mlp_head(joint_features)  # (B, num_joints, 3)
        return pred_joints.reshape(B, -1)


class PTCNNSkeletonModel_Advanced(nn.Module):
    """基于PTCNN的高级骨架模型"""
    def __init__(self, feat_dim: int, output_channels: int, num_joints: int = 22):
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
        
        # 关节点查询与注意力
        self.joint_queries = nn.Parameter(jt.zeros((1, self.num_joints, self.feat_dim)))
        init.gauss_(self.joint_queries, 0, 1)
        self.attention = Attention(embed_dim=self.feat_dim, num_heads=8)
        
        # 坐标回归头
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 3)
        )

    def execute(self, vertices: jt.Var):
        # 输入维度已保证为(B, 3, N)
        xyz = vertices.transpose(0, 2, 1)  # (B, N, 3)
        B, _, N = vertices.shape
        
        # PTCNN特征提取
        x = self.relu(self.bn1(self.conv1(vertices)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat = x.permute(0, 2, 1)
        
        # 分层局部卷积+自注意力
        new_xyz_1, new_feat_1 = sample_and_group(512, 32, xyz, feat)
        feat0 = self.gather0(new_feat_1)
        feat0 = self.sa0(feat0, new_xyz_1.permute(0, 2, 1))
        
        new_xyz_2, new_feat_2 = sample_and_group(256, 32, new_xyz_1, feat0.permute(0, 2, 1))
        feat1 = self.gather1(new_feat_2)
        feat1 = self.sa1(feat1, new_xyz_2.permute(0, 2, 1))
        
        # Point-Transformer深层处理
        pt_feat = self.pt_last(feat1, new_xyz_2)
        
        # 特征融合
        x_fused = concat([pt_feat, feat1], dim=1)
        point_features = self.conv_fuse(x_fused)  # (B, feat_dim, 256)
        
        # 交叉注意力与坐标回归
        point_features_t = point_features.transpose(0, 2, 1)  # (B, 256, feat_dim)
        queries = self.joint_queries.repeat(B, 1, 1)  # (B, num_joints, feat_dim)
        joint_features = self.attention(queries, point_features_t, point_features_t)  # (B, num_joints, feat_dim)
        pred_joints = self.mlp_head(joint_features).reshape(B, -1)
        return pred_joints


# 基础骨架模型变体
class SimpleSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


class SimpleSkeletonModel2(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        self.transformer = Point_Transformer2(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


class SimpleSkeletonModel3(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        self.transformer = Point_Transformer_v3(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


class PTCNNSkeletonModel(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        self.transformer = PTCNN(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


class SimpleSkeletonModellast(nn.Module):
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_channels = output_channels
        self.transformer = Point_Transformer_Last(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)


# 骨架模型工厂函数
def create_model(model_name='ptcnn', output_channels=66, **kwargs):
    num_joints = output_channels // 3
    if model_name == "ptcnn":
        return PTCNNSkeletonModel(feat_dim=256, output_channels=output_channels)
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    if model_name == "pct2":
        return SimpleSkeletonModel2(feat_dim=256, output_channels=output_channels)
    if model_name == "pct3":
        return SimpleSkeletonModel3(feat_dim=256, output_channels=output_channels)
    if model_name == "pctlast":
        return SimpleSkeletonModellast(feat_dim=256, output_channels=output_channels)
    if model_name == "adv":
        return AdvancedSkeletonModel(feat_dim=256, output_channels=output_channels)
    if model_name == "ptcnn_adv":
        return PTCNNSkeletonModel_Advanced(feat_dim=1024, output_channels=output_channels, num_joints=num_joints)
    raise NotImplementedError(f"未实现的模型: {model_name}")