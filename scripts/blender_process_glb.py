"""
    Script of pre-processing GLB meshes via Blender
    NOTE: Blender must be installed first!

    Utils:
        `
        blender --background --factory-startup \
            --python scripts/blender_process_glb.py -- \
                <path_to_glb> \
                <obj_dir>
        `

"""

import sys
import pathlib

import numpy as np
from mathutils import Vector, Matrix

try:
    import bpy
except ImportError:
    print("=> Please install Blender and add it to the PATH")

def normalize(vector):
    return vector / (np.linalg.norm(vector) + 1e-8)


def center_and_rescale():
    all_verts = []
    for object in bpy.context.scene.objects:
        if object.name not in ["Camera", "Light"] and object.data is not None:
            vertices = object.data.vertices
            verts = [object.matrix_world @ vert.co for vert in vertices]
            np_verts = np.array([vert.to_tuple() for vert in verts])
            all_verts.append(np_verts)

    all_verts = np.concatenate(all_verts, axis=0)
    mean_position = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2
    bounds = all_verts.max(axis=0) - all_verts.min(axis=0)
    scale_factor = 1 / bounds.max()

    for object in bpy.context.scene.objects:
        if object.name not in ["Camera", "Light"] and object.data is not None:
            for vert in object.data.vertices:
                non_inverted_new_position = (object.matrix_world @ vert.co - Vector((mean_position[0], mean_position[1], mean_position[2])))
                non_inverted_new_position = Vector((scale_factor, scale_factor, scale_factor)) * non_inverted_new_position
                new_position = object.matrix_world.inverted() @ non_inverted_new_position
                vert.co = new_position

bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()
bpy.data.objects['Light'].select_set(True)
bpy.ops.object.delete()

sys.argv = sys.argv[sys.argv.index("--python") + 2:]
path_to_glb = pathlib.Path(sys.argv[1])
obj_dir = pathlib.Path(sys.argv[2])

print("=> Loading and preprocessing objects")
bpy.ops.import_scene.gltf(filepath=str(path_to_glb))
center_and_rescale()

output_dir = obj_dir / path_to_glb.stem
output_dir.mkdir(exist_ok=True)
bpy.ops.export_scene.obj(filepath=str(output_dir / "mesh.obj"))
