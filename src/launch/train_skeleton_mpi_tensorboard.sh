# 205/6/19 lhz: 分数35
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9" \
# mpirun -np 10 -x PATH -x LD_LIBRARY_PATH \
# python train_skeleton_mpi_tensorboard.py \
#     --train_data_list data/train_list.txt \
#     --val_data_list data/val_list.txt \
#     --batch_size 320 \
#     --symmetry_type position \
#     --symmetry_weight 1

CUDA_VISIBLE_DEVICES="0,1,2,3,8" \
mpirun -np 5 -x PATH -x LD_LIBRARY_PATH \
python train_skeleton_mpi_tensorboard.py \
    --train_data_list data/train_list.txt \
    --val_data_list data/val_list.txt \
    --batch_size 80 \
    --symmetry_type position \
    --symmetry_weight 0 \
    --model_name ptcnn_adv \
