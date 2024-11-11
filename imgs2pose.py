import cupy as cp  # Replaces numpy with cupy for GPU-based operations
import os
import sys
import cv2  # Use OpenCV for GPU-accelerated image processing
import argparse

from colmap_wrapper import run_colmap
import read_write_model as read_model


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = cp.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = cp.array([0, 0, 0, 1.0]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = cp.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = cp.array(im.tvec).reshape([3, 1])
        m = cp.concatenate([cp.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = cp.stack(w2c_mats, 0)
    c2w_mats = cp.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = cp.concatenate([poses, cp.tile(hwf[..., cp.newaxis], [1, 1, poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3D_binary(points3dfile)

    poses = cp.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: cannot access correct camera poses for points')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = cp.array(pts_arr)
    vis_arr = cp.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = cp.sum(-(pts_arr[:, cp.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = cp.percentile(zs, 0.1), cp.percentile(zs, 99.9)

        save_arr.append(cp.concatenate([poses[..., i].ravel(), cp.array([close_depth, inf_depth])], 0))
    save_arr = cp.array(save_arr)

    cp.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith(('JPG', 'jpg', 'png', 'jpeg', 'PNG'))]
    imgs = [cv2.imread(img).astype(cp.float32) / 255. for img in imgs]

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            imgs_down = [cv2.resize(img.get(), (img.shape[1] // r, img.shape[0] // r), interpolation=cv2.INTER_AREA) for img in imgs]
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            imgs_down = [cv2.resize(img.get(), (r[1], r[0]), interpolation=cv2.INTER_AREA) for img in imgs]

        imgdir = os.path.join(basedir, name)
        os.makedirs(imgdir, exist_ok=True)
        for i, img in enumerate(imgs_down):
            cv2.imwrite(os.path.join(imgdir, f'image{i:03d}.png'), (img * 255).astype(cp.uint8).get())


def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = cp.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith(('JPG', 'jpg', 'png'))][0]
    sh = cv2.imread(img0).shape

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith(('JPG', 'jpg', 'png'))]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = cv2.imread(imgfiles[0]).shape
    poses[:2, 4, :] = cp.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    imgs = [cv2.imread(f)[..., :3].astype(cp.float32) / 255. for f in imgfiles]
    imgs = cp.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def gen_poses(basedir, match_type, factors=None):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, match_type)
    else:
        print("Don't need to run COLMAP")

    print('Post-colmap')

    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    if factors is not None:
        print('Factors:', factors)
        minify(basedir, factors)

    print('Done with imgs2poses')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_type', type=str,
                        default='exhaustive_matcher', help='type of matcher used. Valid options: \
                        exhaustive_matcher sequential_matcher. Other matchers not supported at this time')
    parser.add_argument('--data_dir', type=str,
                        help='input scene directory')
    args = parser.parse_args()

    if args.match_type not in ['exhaustive_matcher', 'sequential_matcher']:
        print('ERROR: matcher type ' + args.match_type + ' is not valid. Aborting')
        sys.exit()
    gen_poses(args.data_dir, args.match_type)

