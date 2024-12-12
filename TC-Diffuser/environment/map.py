import torch
import numpy as np
from dataset.homography_warper import get_rotation_matrix2d, warp_affine_crop
#get_rotation_matrix2d是一个图像处理函数 可是数据集只有文本 哪来的图像呢

class Map(object):
    def __init__(self, data, homography, description=None):
        self.data = data
        self.homography = homography
        self.description = description

    def as_image(self):
        raise NotImplementedError

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y, device):
        if self._last_padding == (padding_x, padding_y):
            return self._last_padded_map
        else:
            self._last_padding = (padding_x, padding_y)
            self._last_padded_map = torch.full((self.data.shape[0],
                                                self.data.shape[1] + 2 * padding_x,
                                                self.data.shape[2] + 2 * padding_y),
                                               False, dtype=torch.uint8)
            self._last_padded_map[..., padding_x:-padding_x, padding_y:-padding_y] = self.torch_map(device)
            return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), padding_mode='zeros')

        return rotated_map_batched

    # 这是一个类方法
    # `get_cropped_maps_from_scene_map_batch`，用于从场景地图中提取旋转后的图像补丁。
    #
    # 函数接受以下参数：
    # - `maps`：地理地图对象的列表，表示多个地图（大小为bs）。
    # - `scene_pts`：场景点坐标，形状为[bs, 2]，表示场景中的点。
    # - `patch_size`：旋转后提取的补丁大小，格式为[-x, -y, +x, +y]。
    # - `rotation`：旋转角度（以度为单位），形状为[bs]，可选参数。
    # - `device`：返回旋转后的张量所在的设备。
    #
    # 函数的主要步骤如下：
    # 1.
    # 计算补丁的大小，根据
    # `patch_size`
    # 获取横向和纵向的补丁大小，并确保它们都是偶数。
    # 2.
    # 根据
    # `scene_pts`
    # 和地图的转换关系，计算旋转中心点坐标。
    # 3.
    # 获取地图的填充版本，通过将地图沿着旋转中心进行填充。
    # 4.
    # 根据旋转中心点和旋转角度，将地图进行批量旋转。
    # 5.
    # 删除不再需要的填充地图。
    # 6.
    # 返回提取的旋转和裁剪后的图像补丁。
    #
    # 需要注意的是，函数中涉及到的地理地图对象（GeometricMap）和旋转操作（`batch_rotate`）可能是根据具体应用场景或类的实现定义的，需要结合具体代码和类的实现来理解其完整含义。
    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of each map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        batch_size = scene_pts.shape[0]
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        assert lat_size % 2 == 0, "Patch width must be divisible by 2"
        assert long_size % 2 == 0, "Patch length must be divisible by 2"
        lat_size_half = lat_size // 2
        long_size_half = long_size // 2

        context_padding_x = int(np.ceil(np.sqrt(2) * lat_size))
        context_padding_y = int(np.ceil(np.sqrt(2) * long_size))

        centers = torch.tensor([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],
                               dtype=torch.long, device=device).squeeze(dim=1) \
                  + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)

        padded_map = [s_map.get_padded_map(context_padding_x, context_padding_y, device=device) for s_map in maps]

        padded_map_batched = torch.stack([padded_map[i][...,
                                          centers[i, 0] - context_padding_x: centers[i, 0] + context_padding_x,
                                          centers[i, 1] - context_padding_y: centers[i, 1] + context_padding_y]
                                          for i in range(centers.shape[0])], dim=0)

        center_patches = torch.tensor([[context_padding_y, context_padding_x]],
                                      dtype=torch.int,
                                      device=device).repeat(batch_size, 1)

        if rotation is not None:
            angles = torch.Tensor(rotation)
        else:
            angles = torch.zeros(batch_size)

        rotated_map_batched = cls.batch_rotate(padded_map_batched/255.,
                                                center_patches.float(),
                                                angles,
                                                long_size,
                                                lat_size)

        del padded_map_batched

        return rotated_map_batched[...,
               long_size_half - patch_size[1]:(long_size_half + patch_size[3]),
               lat_size_half - patch_size[0]:(lat_size_half + patch_size[2])]

    def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of the map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, rotation=rotation, device=device)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) > 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points


class ImageMap(Map):  # TODO Implement for image maps -> watch flipped coordinate system
    def __init__(self):
        raise NotImplementedError