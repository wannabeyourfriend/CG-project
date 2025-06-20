CUDA_VISIBLE_DEVICES="0,1,2,3" \
mpirun -np 10 -x PATH -x LD_LIBRARY_PATH \
python train_skeleton_mpi.py \
    --train_data_list data/train_list.txt \
    --val_data_list data/val_list.txt \
    --batch_size 320 \
    --symmetry_type structure \
    --symmetry_weight 0.3 