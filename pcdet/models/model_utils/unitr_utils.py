import torch
import torch.nn as nn
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import three_nn

def get_points(pc_range, sample_num, space_shape, coords=None):
    '''Generate points in specified range or voxels

    Args:
        pc_range (list(int)): point cloud range, (x1,y1,z1,x2,y2,z2)
        sample_num (int): sample point number in a voxel
        space_shape (list(int)): voxel grid shape, (w,h,d)
        coords (tensor): generate points in specified voxels, (N,3)

    Returns:
        points (tensor): generated points, (N,sample_num,3)
    '''
    sx, sy, sz = space_shape
    x1, y1, z1, x2, y2, z2 = pc_range
    if coords is None:
        coord_x = torch.linspace(
            0, sx-1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
        coord_y = torch.linspace(
            0, sy-1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
        coord_z = torch.linspace(
            0, sz-1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
        coords = torch.stack((coord_x, coord_y, coord_z), -1).view(-1, 3)
    points = coords.clone().float()
    points[..., 0] = ((points[..., 0]+0.5)/sx)*(x2-x1) + x1
    points[..., 1] = ((points[..., 1]+0.5)/sy)*(y2-y1) + y1
    points[..., 2] = ((points[..., 2]+0.5)/sz)*(z2-z1) + z1

    if sample_num == 1:
        points = points.unsqueeze(1)
    else:
        points = points.unsqueeze(1).repeat(1, sample_num, 1)
        points[..., 2] = torch.linspace(z1, z2, sample_num).unsqueeze(0)
    return points


def map_points(points, lidar2image, image_aug_matrix, batch_size, image_shape):
    '''Map 3D points to image space.

    Args:
        points (tensor): Grid points in 3D space, shape (grid num, sample num,4).
        lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
        image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
        batch_size (int): Sample number in a batch.
        image_shape (tuple(int)): Image shape, (height, width).

    Returns:
        points (tensor): 3d coordinates of points mapped in image space. shape (B,N,num,k,4)
        points_2d (tensor): 2d coordinates of points mapped in image space. (B,N,num,k,2)
        map_mask (tensor): Number of points per view (batch size x view num). (B,N,num,k,1)
    '''
    points = points.to(torch.float32)
    lidar2image = lidar2image.to(torch.float32)
    image_aug_matrix = image_aug_matrix.to(torch.float32)

    num_view = lidar2image.shape[1]
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    # map points from lidar to (aug) image space
    points = points.unsqueeze(0).unsqueeze(0).repeat(
        batch_size, num_view, 1, 1, 1).unsqueeze(-1)
    grid_num, sample_num = points.shape[2:4]
    lidar2image = lidar2image.view(batch_size, num_view, 1, 1, 4, 4).repeat(
        1, 1, grid_num, sample_num, 1, 1)
    image_aug_matrix = image_aug_matrix.view(
        batch_size, num_view, 1, 1, 4, 4).repeat(1, 1, grid_num, sample_num, 1, 1)
    points_2d = torch.matmul(lidar2image, points).squeeze(-1)

    # recover image augmentation
    eps = 1e-5
    map_mask = (points_2d[..., 2:3] > eps)
    points_2d[..., 0:2] = points_2d[..., 0:2] / torch.maximum(
        points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps)
    points_2d[..., 2] = torch.ones_like(points_2d[..., 2])
    points_2d = torch.matmul(
        image_aug_matrix, points_2d.unsqueeze(-1)).squeeze(-1)[..., 0:2]
    points_2d[..., 0] /= image_shape[1]
    points_2d[..., 1] /= image_shape[0]

    # mask points out of range
    map_mask = (map_mask & (points_2d[..., 1:2] > 0.0)
                & (points_2d[..., 1:2] < 1.0)
                & (points_2d[..., 0:1] < 1.0)
                & (points_2d[..., 0:1] > 0.0))
    map_mask = torch.nan_to_num(map_mask).squeeze(-1)

    return points.squeeze(-1), points_2d, map_mask


class MapImage2Lidar(nn.Module):
    '''Map image patch to lidar space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]

        self.points = get_points(
            self.pc_range, self.sample_num, self.space_shape).cuda()
        self.accelerate = accelerate
        if self.accelerate:
            self.cache = None
        self.use_map = use_map

    def forward(self, batch_dict):
        '''Get the coordinates of image patch in 3D space.

        Returns:
            image2lidar_coords_zyx (tensor): The coordinates of image features 
            (batch size x view num) in 3D space.
            nearest_dist (tensor): The distance between each image feature 
            and the nearest mapped 3d grid point in image space.
        '''

        # accelerate by caching when the mapping relationship changes little
        if self.accelerate and self.cache is not None:
            image2lidar_coords_zyx, nearest_dist = self.cache
            return image2lidar_coords_zyx, nearest_dist
        img = batch_dict['camera_imgs']
        batch_size, num_view, _, h, w = img.shape
        points = self.points.clone()
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']

        with torch.no_grad():
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            points_3d, points_2d, map_mask = map_points(
                points, lidar2image, image_aug_matrix, batch_size, (h, w))
            mapped_points_2d = points_2d[map_mask]
            mapped_points_3d = points_3d[map_mask]
            mapped_view_cnts = map_mask.view(
                batch_size, num_view, -1).sum(-1).view(-1).int()
            mapped_points = torch.cat(
                [mapped_points_2d, torch.zeros_like(mapped_points_2d[:, :1])], dim=-1)
            mapped_coords_3d = mapped_points_3d[:, :3]

            # shape (H*W,2), [[x1,y1],...]
            patch_coords_perimage = batch_dict['patch_coords'][batch_dict['patch_coords'][:, 0] == 0, 2:].clone(
            ).float()
            patch_coords_perimage[:, 0] = (
                patch_coords_perimage[:, 0] + 0.5) / batch_dict['hw_shape'][1]
            patch_coords_perimage[:, 1] = (
                patch_coords_perimage[:, 1] + 0.5) / batch_dict['hw_shape'][0]

            # get image patch coords
            patch_points = patch_coords_perimage.unsqueeze(
                0).repeat(batch_size * num_view, 1, 1).view(-1, 2)
            patch_points = torch.cat(
                [patch_points, torch.zeros_like(patch_points[:, :1])], dim=-1)
            patch_view_cnts = (torch.ones_like(
                mapped_view_cnts) * (batch_dict['hw_shape'][0] * batch_dict['hw_shape'][1])).int()

            # find the nearest 3 mapping points and keep the closest
            _, idx = three_nn(patch_points.to(torch.float32), patch_view_cnts, mapped_points.to(
                torch.float32), mapped_view_cnts)
            idx = idx[:, :1].repeat(1, 3).long()
            # take 3d coords of the nearest mapped point of each image patch as its 3d coords
            image2lidar_coords_xyz = torch.gather(mapped_coords_3d, 0, idx)

            # calculate distance between each image patch and the nearest mapping point in image space
            neighbor_2d = torch.gather(mapped_points, 0, idx)
            nearest_dist = (patch_points[:, :2]-neighbor_2d[:, :2]).abs()
            nearest_dist[:, 0] *= batch_dict['hw_shape'][1]
            nearest_dist[:, 1] *= batch_dict['hw_shape'][0]

            # 3d coords -> voxel grids
            image2lidar_coords_xyz[..., 0] = (image2lidar_coords_xyz[..., 0] - self.pc_range[0]) / (
                self.pc_range[3]-self.pc_range[0]) * self.space_shape[0] - 0.5
            image2lidar_coords_xyz[..., 1] = (image2lidar_coords_xyz[..., 1] - self.pc_range[1]) / (
                self.pc_range[4]-self.pc_range[1]) * self.space_shape[1] - 0.5
            image2lidar_coords_xyz[..., 2] = 0.

            image2lidar_coords_xyz[..., 0] = torch.clamp(
                image2lidar_coords_xyz[..., 0], min=0, max=self.space_shape[0]-1)
            image2lidar_coords_xyz[..., 1] = torch.clamp(
                image2lidar_coords_xyz[..., 1], min=0, max=self.space_shape[1]-1)

            # reorder to z,y,x
            image2lidar_coords_zyx = image2lidar_coords_xyz[:, [2, 1, 0]]
        if self.accelerate:
            self.cache = (image2lidar_coords_zyx, nearest_dist)
        return image2lidar_coords_zyx, nearest_dist


class MapLidar2Image(nn.Module):
    '''Map Lidar points to image space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]
        self.accelerate = accelerate
        if self.accelerate:
            self.full_lidar2image_coors_zyx = None
            # only support one point in a voxel
            self.points = get_points(
                self.pc_range, self.sample_num, self.space_shape).cuda()
        self.use_map = use_map

    def pre_compute(self, batch_dict):
        '''Precalculate the coords of all voxels mapped on the image'''
        image = batch_dict['camera_imgs']
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        image_shape = image.shape[-2:]
        assert image.shape[0] == 1, 'batch size should be 1 in pre compute'
        batch_idx = torch.zeros(
            self.space_shape[0]*self.space_shape[1], device=image.device)
        with torch.no_grad():
            # get reference points, only in voxels.
            points = self.points.clone()
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, image_aug_matrix, batch_idx, image_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            self.full_lidar2image_coors_zyx = lidar2image_coords_xyz[:, [
                2, 0, 1]]

    def map_lidar2image(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape):
        '''Map Lidar points to image space.

        Args:
            points (tensor): batch lidar points shape (voxel num, sample num,4).
            lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
            image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
            batch_idx (tensor): batch id for all points in batch
            image_shape (Tuple(int, int)): Image shape, (height, width).

        Returns:
            batch_hit_points: 2d coordinates of lidar points mapped in image space. 
        '''
        num_view = lidar2image.shape[1]
        batch_size = (batch_idx[-1] + 1).int()
        batch_hit_points = []
        for b in range(batch_size):
            _, points_2d, map_mask = map_points(
                points[batch_idx == b], lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape)
            points_2d = points_2d.squeeze(3)
            # set point not hit image as hit 0
            map_mask = map_mask.squeeze(3).permute(0, 2, 1).view(-1, num_view)
            hit_mask = map_mask.any(dim=-1)
            map_mask[~hit_mask, 0] = True
            # get hit view id
            hit_view_ids = torch.nonzero(map_mask)
            # select first view if hit multi view
            hit_poins_id = hit_view_ids[:, 0]
            shift_hit_points_id = torch.roll(hit_poins_id, 1)
            shift_hit_points_id[0] = -1
            first_mask = (hit_poins_id - shift_hit_points_id) > 0
            unique_hit_view_ids = hit_view_ids[first_mask, 1:]
            num = points_2d.shape[2]
            assert len(unique_hit_view_ids) == num, 'some points not hit view!'
            # get coords in hit view
            points_2d = points_2d.permute(0, 2, 1, 3).flatten(0, 1)
            hit_points_2d = points_2d[range(
                num), unique_hit_view_ids.squeeze()]
            # clamp value range and adjust to postive for set partition
            hit_points_2d = torch.clamp(hit_points_2d, -1, 2) + 1
            hit_points = torch.cat([hit_points_2d, unique_hit_view_ids], -1)
            batch_hit_points.append(hit_points)
        batch_hit_points = torch.cat(batch_hit_points, dim=0)
        return batch_hit_points

    def forward(self, batch_dict):
        '''Get the coordinates of lidar poins in image space.

        Returns:
            lidar2image_coords_zyx (tensor): The coordinates of lidar points in 3D space.
        '''
        if self.accelerate:
            if self.full_lidar2image_coors_zyx is None:
                self.pre_compute(batch_dict)
            # accelerate by index table
            coords_xyz = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone()
            unique_index = coords_xyz[:, 1] * \
                self.space_shape[1] + coords_xyz[:, 2]
            lidar2image_coords_zyx = self.full_lidar2image_coors_zyx[unique_index.long(
            )]
            return lidar2image_coords_zyx
        img = batch_dict['camera_imgs']
        coords = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone()
        lidar2image = batch_dict['lidar2image']
        img_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        img_shape = img.shape[-2:]
        batch_idx = coords[:, 0]
        with torch.no_grad():
            # get reference points, only in voxels.
            points = get_points(self.pc_range, self.sample_num,
                                self.space_shape, coords[:, 1:])
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, img_aug_matrix, batch_idx, img_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            lidar2image_coords_zyx = lidar2image_coords_xyz[:, [2, 0, 1]]

        return lidar2image_coords_zyx
