import argparse
import numpy as np
import os
from random import shuffle

from dataset.asset import Asset
from dataset.exporter import Exporter
from dataset.format import parents, id_to_name


def main(args):
    '''
    Debug example. Reads one asset, one animation track and binds them and then export as fbx format.
    Users can visualize the results in softwares like blender.
    '''
    render = args.render
    export_fbx = args.export_fbx
    predict_output_dir = args.predict_output_dir

    if export_fbx is True:
        try:
            import bpy
        except:
            print("need bpy==4.2 & python==3.11")
            return

    exporter = Exporter()
    roots = []
    for root, dirs, files in os.walk(predict_output_dir):
        if 'predict_skeleton.npy' in files and 'predict_skin.npy' in files:
            roots.append(root)
    shuffle(roots)
    for root in roots:
        print(f"export {os.path.relpath(root, predict_output_dir)}")
        save_path = os.path.join(args.render_output_dir, os.path.relpath(root, predict_output_dir))
        os.makedirs(save_path, exist_ok=True)
        vertices = np.load(os.path.join(root, 'transformed_vertices.npy'))
        skin = np.load(os.path.join(root, 'predict_skin.npy'))
        joints = np.load(os.path.join(root, 'predict_skeleton.npy'))
        if render:
            exporter._render_skeleton(
                path=os.path.join(save_path, 'skeleton.png'),
                joints=joints,
                parents=parents,
            )
            for id in id_to_name:
                name = id_to_name[id]
                exporter._render_skin(
                    path=os.path.join(save_path, f'skin_{name}.png'),
                    skin=skin[:, id],
                    vertices=vertices,
                    joint=joints[id],
                )
        if export_fbx:
            names = [f'bone_{i}' for i in range(len(parents))]
            asset = Asset.load(path=os.path.join("data", root.replace('predict', 'test')+".npz"))
            exporter._export_fbx(
                path=os.path.join(save_path, f'res.fbx'),
                vertices=vertices,
                joints=joints,
                skin=skin,
                parents=parents,
                names=names,
                faces=asset.faces,
            )

def str2bool(val):
    val = val.lower()
    if val == 'false':
        return False
    elif val == 'true':
        return True
    else:
        raise NotImplementedError(f"expect false or true, found {val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_output_dir", type=str, required=True)
    parser.add_argument("--render_output_dir", type=str, required=True)
    parser.add_argument("--render", type=str2bool, required=False, default=True)
    parser.add_argument("--export_fbx", type=str2bool, required=False, default=False)
    
    args = parser.parse_args()
    main(args)
