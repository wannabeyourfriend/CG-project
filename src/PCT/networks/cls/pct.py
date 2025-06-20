import jittor as jt
from jittor import nn  
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    # xyz = xyz.contiguous()
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) # [B, npoint]
    # print ('fps size=', fps_idx.size())
    # fps_idx = sampler(xyz).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points



class Point_Transformer2(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer2, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = Point_Transformer_Last()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def execute(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        # add position embedding on each layer
        x = self.pt_last(feature_1, new_xyz)
        x = concat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x



class Point_Transformer(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        # x is expected to be [B, 3, N]
        batch_size, C, N = x.size()
        
        # Store original input for xyz coordinates
        x_input = x
        
        # Apply convolutions
        x = self.relu(self.bn1(self.conv1(x)))  # B, 128, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # Apply self-attention layers with xyz coordinates
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)
        x3 = self.sa3(x2, x_input)
        x4 = self.sa4(x3, x_input)
        
        # Concatenate features from all SA layers
        x = concat((x1, x2, x3, x4), dim=1)

        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
    def execute(self, x, xyz):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        # xyz = self.conv_pos(xyz) 删除错误的代码
        # end
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        
        x = concat((x1, x2, x3, x4), dim=1)

        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def execute(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

# 添加 Point_Transformer_v3 类
class Point_Transformer_v3(nn.Module):
    def __init__(self, output_channels=40):
        super(Point_Transformer_v3, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(256)
        self.sa4 = SA_Layer(256)

        self.conv_transition1 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn_transition1 = nn.BatchNorm1d(256)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(768, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2)
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()
        
    def execute(self, x):
        batch_size, C, N = x.size()
        x_input = x  # Save for positional encoding

        x = self.relu(self.bn1(self.conv1(x)))  # B, 64, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, 128, N

        # First two SA layers with 128 channels
        x1 = self.sa1(x, x_input)
        x2 = self.sa2(x1, x_input)

        # Transition to higher dimensionality
        x_transitioned = self.relu(self.bn_transition1(self.conv_transition1(x2)))  # B, 256, N

        # Third and fourth SA layers with 256 channels
        x3 = self.sa3(x_transitioned, x_input)
        x4 = self.sa4(x3, x_input)

        # Concatenate features
        x_cat = concat((x1, x2, x3, x4), dim=1)  # B, 768, N

        x = self.conv_fuse(x_cat)
        x = jt.max(x, 2)
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class PTCNN(nn.Module):
    """
    Point Transformer + CNN (PTCNN)

    1. 两层 1 × 1 卷积提取初始几何特征（64 → 64）。
    2. 两级 “采样-分组-局部卷积” 提升到 128 / 256 维，并在每级之后
       追加一个自注意力 SA_Layer 以增强局部关联。
    3. Point_Transformer_Last 进一步堆叠 4 个 SA_Layer，
       获得 1024 维全局上下文特征并与 256 维局部特征拼接（1280 维）。
    4. 1 × 1 卷积压缩到 1024 维后做全局最大池化，接全连接分类头。

    输入:
        x: 形状 [B, 3, N] 的点云坐标 (float32)

    输出:
        形状 [B, output_channels] 的分类 logits
    """
    def __init__(self, output_channels: int = 40):
        super(PTCNN, self).__init__()

        # (1) 低阶几何特征
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(64)

        # (2) 分层局部卷积 + 自注意力
        self.gather0 = Local_op(128, 128)   # in/out = 128
        self.sa0     = SA_Layer(128)

        self.gather1 = Local_op(256, 256)   # in/out = 256
        self.sa1     = SA_Layer(256)

        # (3) 深层 Point-Transformer（4×SA）
        self.pt_last = Point_Transformer_Last()  # 默认 channels=256

        # (4) 特征融合与分类
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2),
        )

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, output_channels)
        self.relu = nn.ReLU()

    def execute(self, x):          # x: [B, 3, N]
        xyz = x.permute(0, 2, 1)   # [B, N, 3]
        B, _, _ = x.size()

        # (1) 初始几何特征
        x = self.relu(self.bn1(self.conv1(x)))    # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))    # [B, 64, N]
        feat = x.permute(0, 2, 1)                 # [B, N, 64]

        # (2-a) 第 1 级局部卷积
        new_xyz, new_feat = sample_and_group(512, 32, xyz, feat)   # [B,512,3] & [B,512,32,64+3]
        feat0 = self.gather0(new_feat)                             # [B,128,512]
        feat0 = self.sa0(feat0, new_xyz.permute(0, 2, 1))          # 加注意力

        # (2-b) 第 2 级局部卷积
        new_xyz, new_feat = sample_and_group(256, 32,
                                             new_xyz, feat0.permute(0, 2, 1))
        feat1 = self.gather1(new_feat)                             # [B,256,256]
        feat1 = self.sa1(feat1, new_xyz.permute(0, 2, 1))          # 加注意力

        # (3) Point-Transformer 深层堆叠
        pt_feat = self.pt_last(feat1, new_xyz)                     # [B,1024,256]

        # (4) 全局融合与分类
        x = concat([pt_feat, feat1], dim=1)        # [B,1280,256]
        x = self.conv_fuse(x)                      # [B,1024,256]
        x = jt.max(x, 2)                           # 全局 max-pool, [B,1024]
        x = x.view(B, -1)

        x = self.relu(self.bn6(self.fc1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)                            # [B, output_channels]
        return x




class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
      # self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # Add a projection for xyz coordinates
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)

    def execute(self, x, xyz):
        # Project xyz to the same channel dimension as x
        xyz_feat = self.xyz_proj(xyz)
        
        # Now we can safely add them
        x = x + xyz_feat
        
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = nn.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = nn.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

if __name__ == '__main__':
    
    jt.flags.use_cuda=1
    input_points = init.gauss((16, 3, 1024), dtype='float32')  # B, D, N 


    network = Point_Transformer()
    out_logits = network(input_points)
    print (out_logits.shape)

