# Batch Size 为 320 时候, 很早就陷入局部极值,()
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9" mpirun -np 10 -x PATH -x LD_LIBRARY_PATH python train_skin_mpi_tensorboard.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --batch_size 640
# Batch Size 为 160 时候, 70 EPOCH
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9" mpirun -np 10 -x PATH -x LD_LIBRARY_PATH python train_skin_mpi_tensorboard.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --batch_size 320
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9" mpirun -np 10 -x PATH -x LD_LIBRARY_PATH python train_skin_mpi_tensorboard.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --batch_size 80 --model_name ptcnn_adv
