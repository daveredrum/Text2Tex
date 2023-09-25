import xatlas
import trimesh
import argparse

def parameterize_mesh(input_path, output_path):
    # parameterize the mesh
    mesh = trimesh.load_mesh(input_path, force='mesh')
    vertices, faces = mesh.vertices, mesh.faces

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    parameterize_mesh(args.input_path, args.output_path)
    