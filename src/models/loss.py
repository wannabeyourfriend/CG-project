# 2025/6/18 by Wang Zixuan
import jittor as jt
from jittor import nn

class ChamferDistanceLoss(nn.Module):
    def execute(self, pred_points: jt.Var, gt_points: jt.Var) -> jt.Var:
        if pred_points.ndim == 2:
            pred_points = pred_points.reshape(pred_points.shape[0], -1, 3)
        if gt_points.ndim == 2:
            gt_points = gt_points.reshape(gt_points.shape[0], -1, 3)


        pred_expanded = pred_points.unsqueeze(2)
        gt_expanded = gt_points.unsqueeze(1)
        
        dist_matrix_sq = ((pred_expanded - gt_expanded)**2).sum(dim=-1)
        
        dist_matrix = jt.sqrt(dist_matrix_sq + 1e-12)

        dist_pred_to_gt, _ = dist_matrix.min(dim=2) 
        
        dist_gt_to_pred, _ = dist_matrix.min(dim=1)

        cd_loss = dist_pred_to_gt.mean(dim=1) + dist_gt_to_pred.mean(dim=1)
        
        return cd_loss.mean()

class SymmetricJointLoss(nn.Module):
    '''
    对称关节位置损失：广义的对称
    - none: 不启用对称性损失；
    - position: 直接空间镜像 适用于 T-pose
    - structure: 相对于根节点方向镜像（适用于动态姿态）；
    '''
    def __init__(self, mode='none', joint_pairs=None, root_joint_id=0):
        super().__init__()
        assert mode in ['none', 'position', 'structure'], \
            f"未知对称模式: {mode}"
        self.mode = mode
        self.symmetric_joint_pairs = joint_pairs or [
            (1, 2),#hip
            (7, 8),#shoulder
            (9, 10),#upper arm
            (11, 12),#lower arm
            (13, 14),#hand 
            (15, 16),#upper leg
            (17, 18),#lower leg
            (19, 20)#foot
        ]
        self.root_id = root_joint_id

    def execute(self, joints: jt.Var) -> jt.Var:
        """
        joints: [B, J, 3] 预测关节坐标
        """
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        if self.mode == 'none':
            return jt.zeros(1)

        B = joints.shape[0]
        loss = 0.0
        for left, right in self.symmetric_joint_pairs:
            left_joint = joints[:, left, :] # [B, 3]
            right_joint = joints[:, right, :] # [B, 3]
            root_joint = joints[:, self.root_id, :].unsqueeze(1)  # [B, 1, 3]

            if self.mode == 'position':
                mirrored_right = right_joint.clone()
                mirrored_right[:, 0] = -mirrored_right[:, 0]
                loss += jt.norm(left_joint - mirrored_right, dim=1).mean()
            
            elif self.mode == 'structure':
                left_vector = left_joint - root_joint.squeeze(1)  # [B, 3]
                right_vector = right_joint - root_joint.squeeze(1)  # [B, 3]
                mirrored_right_vector = right_vector.clone()
                mirrored_right_vector[:, 0] = -mirrored_right_vector[:, 0]
                loss += jt.norm(left_vector - right_vector, dim=1).mean()

        return loss / len(self.symmetric_joint_pairs)
    
class SymmetricSkinLoss(nn.Module):
    '''
    对称蒙皮权重损失：用于约束对称点的皮肤权重一致。
    需要传入 symmetry_index 映射。
    '''
    def __init__(self, symmetry_index):
        super().__init__()
        self.symmetry_index = symmetry_index  # [N] numpy or jt.Var

    def execute(self, skin: jt.Var) -> jt.Var:
        """
        skin: [B, N, J]
        self.symmetry_index: [N], 每个点的对称点索引
        """
        mirrored_skin = skin[:, self.symmetry_index, :]  # [B, N, J]
        return jt.abs(skin - mirrored_skin).mean()



class L1Loss(nn.Module):
    def execute(self, pred: jt.Var, gt: jt.Var) -> jt.Var:

        assert pred.shape == gt.shape, "预测值和真实值的形状必须相同"
        return jt.abs(pred - gt).mean()

