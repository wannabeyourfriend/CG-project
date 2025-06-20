CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9" mpirun -np 10 -x PATH -x LD_LIBRARY_PATH python train_skin_mpi.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt --batch_size 640
