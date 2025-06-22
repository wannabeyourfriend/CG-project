# By Wangzixuan 2025/6/20
import jittor as jt
from jittor import nn, init
from jittor.contrib import concat
from math import sqrt

from PCT.networks.cls.pct import SA_Layer, Local_op, sample_and_group, Point_Transformer_Last
from PCT.misc.ops import knn_point, index_points, square_distance, topk

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, query, key, value):
        B, N_q, C = query.shape
        B, N_k, C_k = key.shape
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = jt.matmul(q, k) * (self.head_dim ** -0.5)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)
        context = jt.matmul(attn_probs, v).permute(0, 2, 1, 3).reshape(B, N_q, C)
        return self.out_proj(context)

class DifferentiableLBS(nn.Module):
    def __init__(self):
        super(DifferentiableLBS, self).__init__()

    def execute(self, V, W, J_inv_mats, pose):
        batch_size, num_points, _ = V.shape
        T = jt.matmul(pose, J_inv_mats)
        # 修复: nn.concat -> concat
        V_homo = concat([V, jt.ones((batch_size, num_points, 1))], dim=2)
        W_expanded = W.unsqueeze(-1).unsqueeze(-1)
        T_expanded = T.unsqueeze(1)
        G = (W_expanded * T_expanded).sum(dim=2)
        V_homo_reshaped = V_homo.unsqueeze(-1)
        V_posed_homo = jt.matmul(G, V_homo_reshaped)
        V_posed = V_posed_homo.squeeze(-1)[:, :, :3]
        return V_posed


class AdvancedJointModel_v1(nn.Module):
    def __init__(self, num_joints=24, feat_dim=1024, num_keypoints=256):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints

        self.conv1 = nn.Conv1d(3, 64, 1, bias=False); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False); self.bn2 = nn.BatchNorm1d(64)
        self.gather0 = Local_op(128, 128); self.sa0 = SA_Layer(128)
        self.gather1 = Local_op(256, 256); self.sa1 = SA_Layer(256)
        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, self.feat_dim, 1, bias=False),
            nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(scale=0.2),
        )
        self.relu = nn.ReLU()

        self.joint_queries = nn.Parameter(jt.zeros((1, self.num_joints, self.feat_dim)))
        init.gauss_(self.joint_queries, 0, 1)
        self.skeleton_attention = Attention(embed_dim=self.feat_dim, num_heads=8)
        self.skeleton_mlp_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 3)
        )

        self.skin_vertex_feature_mlp = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.ReLU(),
        )
        self.skin_joint_pos_embed = nn.Linear(3, self.feat_dim)
        self.skin_joint_attention = Attention(embed_dim=self.feat_dim, num_heads=8)

    def execute(self, vertices):
        """
        Args:
            vertices (jt.Var): 输入点云, shape [B, N, 3]
        Returns:
            pred_joints (jt.Var): 预测的T-pose骨骼节点, shape [B, J, 3]
            pred_weights (jt.Var): 预测的蒙皮权重, shape [B, N, J]
        """
        B, N, _ = vertices.shape

        x = vertices.transpose(0, 2, 1) # (B, 3, N)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat = x.permute(0, 2, 1) # (B, N, 64)

        xyz_1, feat_1 = sample_and_group(512, 32, vertices, feat)
        feat0 = self.sa0(self.gather0(feat_1), xyz_1.permute(0, 2, 1))
        
        xyz_2, feat_2 = sample_and_group(self.num_keypoints, 32, xyz_1, feat0.permute(0, 2, 1))
        feat1 = self.sa1(self.gather1(feat_2), xyz_2.permute(0, 2, 1))

        pt_feat = self.pt_last(feat1, xyz_2)
        
        fused_feat = concat([pt_feat, feat1], dim=1) # (B, 1024+256, K)
        keypoint_features = self.conv_fuse(fused_feat) # (B, feat_dim, K)
        keypoint_features_t = keypoint_features.transpose(0, 2, 1) # (B, K, feat_dim)
        keypoint_xyz = xyz_2 # (B, K, 3)

        queries = self.joint_queries.repeat(B, 1, 1)
        joint_features = self.skeleton_attention(queries, keypoint_features_t, keypoint_features_t)
        pred_joints = self.skeleton_mlp_head(joint_features) # (B, J, 3)

        sqrdists = square_distance(vertices, keypoint_xyz)
        dist, idx = topk(sqrdists, 3, dim=-1, largest=False)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = jt.sum(dist_recip, dim=2, keepdims=True)
        weight = dist_recip / norm
        interpolated_features = jt.sum(index_points(keypoint_features_t, idx) * weight.view(B, N, 3, 1), dim=2)
        
        vertex_features = self.skin_vertex_feature_mlp(concat([vertices, interpolated_features], dim=-1))

        joint_queries_for_skin = self.skin_joint_pos_embed(pred_joints)
        joint_features_for_skin = self.skin_joint_attention(joint_queries_for_skin, keypoint_features_t, keypoint_features_t)

        attn_logits = vertex_features @ joint_features_for_skin.transpose(0, 2, 1)
        pred_weights = nn.softmax(attn_logits / sqrt(self.feat_dim), dim=-1)

        return pred_joints, pred_weights


class AdvancedJointModel(nn.Module):
    """

    """
    def __init__(self, num_joints=24, feat_dim=1024, num_keypoints=512):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints

        self.conv1 = nn.Conv1d(3, 64, 1, bias=False); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False); self.bn2 = nn.BatchNorm1d(64)
        self.gather0 = Local_op(128, 128); self.sa0 = SA_Layer(128)
        self.gather1 = Local_op(256, 256); self.sa1 = SA_Layer(256)
        self.pt_last = Point_Transformer_Last()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, self.feat_dim, 1, bias=False),
            nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(scale=0.2),
        )
        self.relu = nn.ReLU()

        self.joint_queries = nn.Parameter(jt.zeros((1, self.num_joints, self.feat_dim)))
        init.gauss_(self.joint_queries, 0, 1)
        self.skeleton_attention = Attention(embed_dim=self.feat_dim, num_heads=8)
        self.ln_skel_attn = nn.LayerNorm(self.feat_dim) 
        self.skeleton_mlp_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.LayerNorm(self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 3)
        )

        self.skin_vertex_feature_mlp = nn.Sequential(
            nn.Linear(self.feat_dim + 3, self.feat_dim),
            nn.LayerNorm(self.feat_dim), 
            nn.ReLU(),
        )
        self.skin_joint_pos_embed = nn.Linear(3, self.feat_dim)
        self.skin_joint_attention = Attention(embed_dim=self.feat_dim, num_heads=8)
        self.ln_skin_attn = nn.LayerNorm(self.feat_dim)

    def execute(self, vertices):
        B, N, _ = vertices.shape

        x = vertices.transpose(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat = x.permute(0, 2, 1)

        xyz_1, feat_1 = sample_and_group(512, 32, vertices, feat)
        feat0 = self.sa0(self.gather0(feat_1), xyz_1.permute(0, 2, 1))
        
        xyz_2, feat_2 = sample_and_group(self.num_keypoints, 32, xyz_1, feat0.permute(0, 2, 1))
        feat1 = self.sa1(self.gather1(feat_2), xyz_2.permute(0, 2, 1))

        pt_feat = self.pt_last(feat1, xyz_2)
        
        fused_feat = concat([pt_feat, feat1], dim=1)
        keypoint_features = self.conv_fuse(fused_feat)
        keypoint_features_t = keypoint_features.transpose(0, 2, 1)
        keypoint_xyz = xyz_2

        queries = self.joint_queries.repeat(B, 1, 1)
        attn_out_skel = self.skeleton_attention(queries, keypoint_features_t, keypoint_features_t)
        joint_features = self.ln_skel_attn(queries + attn_out_skel)
        pred_joints = self.skeleton_mlp_head(joint_features)

        sqrdists = square_distance(vertices, keypoint_xyz)
        dist, idx = topk(sqrdists, 3, dim=-1, largest=False)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = jt.sum(dist_recip, dim=2, keepdims=True)
        weight = dist_recip / norm
        interpolated_features = jt.sum(index_points(keypoint_features_t, idx) * weight.view(B, N, 3, 1), dim=2)
        
        vertex_features = self.skin_vertex_feature_mlp(concat([vertices, interpolated_features], dim=-1))

        joint_queries_for_skin = self.skin_joint_pos_embed(pred_joints)
        attn_out_skin = self.skin_joint_attention(joint_queries_for_skin, keypoint_features_t, keypoint_features_t)
        joint_features_for_skin = self.ln_skin_attn(joint_queries_for_skin + attn_out_skin)

        attn_logits = vertex_features @ joint_features_for_skin.transpose(0, 2, 1)
        pred_weights = nn.softmax(attn_logits / sqrt(self.feat_dim), dim=-1)

        return pred_joints, pred_weights