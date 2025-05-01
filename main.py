import sys
import argparse
import subprocess
import importlib
import torch
import cv2
import numpy as np
import open3d as o3d
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename


def ensure(pkg: str, mod: str | None = None) -> None:
    """Import *mod* (or *pkg*). If that fails, pip-install *pkg* then re-import."""
    try:
        importlib.import_module(mod or pkg)
    except ModuleNotFoundError:
        print(f'Installing missing dependency: {pkg}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        importlib.invalidate_caches()
        importlib.import_module(mod or pkg)


# MiDaS relies on “timm” for its backbones
ensure('timm')


def estimate_depth(img: np.ndarray,
                   model: torch.nn.Module,
                   transform,
                   device: torch.device) -> np.ndarray:
    tensor = transform(img).to(device)
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
    mn, mx = pred.min(), pred.max()
    return (pred - mn) / (mx - mn) if mx > mn else np.zeros_like(pred)


def depth_to_point_cloud(depth: np.ndarray, img: np.ndarray) -> o3d.geometry.PointCloud:
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    fx = fy = max(h, w)
    cx, cy = w / 2.0, h / 2.0

    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    return pc


def reconstruct_mesh(pc: o3d.geometry.PointCloud,               # changed
                     depth_start: int = 12,
                     density_thresh: float = 0.05,
                     smooth_iters: int = 30) -> o3d.geometry.TriangleMesh:
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # try Poisson at decreasing depths until it behaves
    for d in range(depth_start, 5, -1):
        try:
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=d)
            if np.isfinite(dens).all():
                break
        except RuntimeError:
            continue
    else:  # Poisson hopeless → fall back to ball-pivoting
        radii = o3d.utility.DoubleVector([0.004, 0.008, 0.016])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, radii)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=60_000)
        mesh.compute_vertex_normals()
        return mesh

    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_thresh))
    mesh = mesh.crop(pc.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    mesh.compute_vertex_normals()
    return mesh



def main() -> None:
    parser = argparse.ArgumentParser(description='Convert an image to a detailed 3D mesh.')
    parser.add_argument('-i', '--input', help='input image path')
    parser.add_argument('-o', '--output', help='output mesh path (.ply or .obj)')
    args = parser.parse_args()

    if args.input and args.output:
        img_path, out_path = args.input, args.output
    else:
        root = Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        print('Select input image…')
        img_path = askopenfilename(
            title='Choose an image',
            filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp')]
        )
        if not img_path:
            sys.exit('No image selected.')
        print('Select output file…')
        out_path = asksaveasfilename(
            title='Save mesh as',
            defaultextension='.ply',
            filetypes=[('Mesh', '*.ply;*.obj')]
        )
        if not out_path:
            sys.exit('No output path provided.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading MiDaS DPT_Large model…')
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)
    midas.to(device).eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).dpt_transform

    print('Reading image…')
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f'Failed to read image: {img_path}')

    print('Estimating depth…')
    depth = estimate_depth(img, midas, transform, device)

    print('Smoothing depth…')
    depth_u8 = cv2.bilateralFilter((depth * 255).astype(np.uint8), 9, 75, 75)
    depth = depth_u8.astype(np.float32) / 255.0

    print('Building point cloud…')
    pc = depth_to_point_cloud(depth, img)

    print('Reconstructing mesh…')
    mesh = reconstruct_mesh(pc)

    print('Displaying mesh…')
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print('Saving mesh…')
    if not o3d.io.write_triangle_mesh(out_path, mesh):
        sys.exit(f'Could not save mesh to: {out_path}')
    print(f'Mesh saved to {out_path}')


if __name__ == '__main__':
    main()
