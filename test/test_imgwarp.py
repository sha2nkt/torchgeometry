import unittest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils


class Tester(unittest.TestCase):

    def test_normalize_pixel_grid(self):
        # generate input data
        batch_size = 1
        height, width = 2, 4

        # create points grid
        grid_norm = tgm.create_meshgrid(height, width,
            normalized_coordinates=True)
        grid_norm = torch.unsqueeze(grid_norm, dim=0)

        grid_pix = tgm.create_meshgrid(height, width,
            normalized_coordinates=False)
        grid_pix = torch.unsqueeze(grid_pix, dim=0)

        # grid from pixel space to normalized
        norm_trans_pix = tgm.normal_transform_pixel(height, width)  # 1x3x3
        pix_trans_norm = tgm.inverse(norm_trans_pix)  # 1x3x3

        # transform grids
        grid_pix_to_norm = tgm.transform_points(norm_trans_pix, grid_pix)
        grid_norm_to_pix = tgm.transform_points(pix_trans_norm, grid_norm)

        self.assertTrue(utils.check_equal_torch(grid_pix, grid_norm_to_pix))
        self.assertTrue(utils.check_equal_torch(grid_norm, grid_pix_to_norm))

    def test_normalize_transform_pixel(self):
        # generate input data
        batch_size = 1
        height, width = 2, 4

        # create transforms a center of the image
        transform_pix = utils.create_eye_batch(batch_size, 3)
        transform_pix[..., 0, 2] = (width - 1) / 2
        transform_pix[..., 1, 2] = (height - 1) / 2

        # create points grid
        grid_norm = tgm.create_meshgrid(height, width,
            normalized_coordinates=True)
        grid_pix = tgm.create_meshgrid(height, width,
            normalized_coordinates=False)

        # transform grid
        grid_pix_transformed = tgm.transform_points(transform_pix, grid_pix)

        # grid from pixel space to normalized
        norm_trans_pix = tgm.normal_transform_pixel(height, width)  # 1x3x3
        grid_pix_transformed_to_norm = tgm.transform_points(
            norm_trans_pix, grid_pix_transformed)

        # top-left pixel should be mapped to image center which in normalized
        # coordinates is (0, 0)
        self.assertTrue(utils.check_equal_torch(
            grid_pix_transformed_to_norm[0, 0, 0], torch.zeros(2)))

        # transform directly normalized grid
        norm_trans_norm = tgm.normalize_transform_to_pix(
            transform_pix, height, width)
        grid_norm_transformed = tgm.transform_points(norm_trans_norm, grid_norm)

        # the pixel grid transformed and then normalized should be equal
        # to the grid normalized after transformed by transform wich converts
        # from norm-pixel-transform-pixel-norm
        self.assertTrue(utils.check_equal_torch(
            grid_pix_transformed_to_norm, grid_norm_transformed))

    def test_warp_perspective(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32
        alpha = tgm.pi / 2  # 90 deg rotation

        # create data patch
        patch = torch.rand(batch_size, 1, height, width)

        # create transformation (rotation)
        M = torch.tensor([[
            [torch.cos(alpha), -torch.sin(alpha), 0.],
            [torch.sin(alpha), torch.cos(alpha), 0.],
            [0., 0., 1.],
        ]])  # Bx3x3

        # apply transformation and inverse
        _, _, h, w = patch.shape
        patch_warped = tgm.warp_perspective(patch, M, dsize=(height, width))
        patch_warped_inv = tgm.warp_perspective(patch_warped, tgm.inverse(M),
                                                dsize=(height, width))

        # generate mask to compute error
        mask = torch.ones_like(patch)
        mask_warped_inv = tgm.warp_perspective(
            tgm.warp_perspective(patch, M, dsize=(height, width)),
            tgm.inverse(M), dsize=(height, width))

        res = utils.check_equal_torch(mask_warped_inv * patch,
                                      mask_warped_inv * patch_warped_inv)
        self.assertTrue(res)

    def test_warp_perspective(self):
        # generate input data
        batch_size = 1
        height, width = 16, 32
        alpha = tgm.pi / 2  # 90 deg rotation

        # create data patch
        patch = torch.rand(batch_size, 1, height, width)

        # create transformation (rotation)
        M = torch.tensor([[
            [torch.cos(alpha), -torch.sin(alpha), 0.],
            [torch.sin(alpha), torch.cos(alpha), 0.],
            [0., 0., 1.],
        ]])  # Bx3x3

        # apply transformation and inverse
        _, _, h, w = patch.shape
        patch_warped = tgm.warp_perspective(patch, M, dsize=(height, width))
        patch_warped_inv = tgm.warp_perspective(patch_warped, tgm.inverse(M),
                                                dsize=(height, width))

        # generate mask to compute error
        mask = torch.ones_like(patch)
        mask_warped_inv = tgm.warp_perspective(
            tgm.warp_perspective(patch, M, dsize=(height, width)),
            tgm.inverse(M), dsize=(height, width))

        res = utils.check_equal_torch(mask_warped_inv * patch,
                                      mask_warped_inv * patch_warped_inv)
        self.assertTrue(res)

    def test_warp_perspective_crop(self):
        # generate input data
        batch_size = 1
        src_h, src_w = 3, 4
        dst_h, dst_w = 3, 2

        # [x, y] origin
        # top-left, top-right, bottom-right, bottom-left
        points_src = torch.FloatTensor([[
            [1, 0], [2, 0], [2, 2], [1, 2],
        ]])

        # [x, y] destination
        # top-left, top-right, bottom-right, bottom-left
        points_dst = torch.FloatTensor([[
            [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1],
        ]])

        # compute transformation between points
        dst_pix_trans_src_pix = tgm.get_perspective_transform(
            points_src, points_dst)

        # create points grid in normalized coordinates
        grid_src_norm = tgm.create_meshgrid(src_h, src_w,
            normalized_coordinates=True)
        grid_src_norm = torch.unsqueeze(grid_src_norm, dim=0)

        # create points grid in pixel coordinates
        grid_src_pix = tgm.create_meshgrid(src_h, src_w,
            normalized_coordinates=False)
        grid_src_pix = torch.unsqueeze(grid_src_pix, dim=0)

        src_norm_trans_src_pix = tgm.normal_transform_pixel(src_h, src_w)
        src_pix_trans_src_norm = tgm.inverse(src_norm_trans_src_pix)

        dst_norm_trans_dst_pix = tgm.normal_transform_pixel(dst_h, dst_w)

        # transform pixel grid
        grid_dst_pix = tgm.transform_points(
            dst_pix_trans_src_pix, grid_src_pix)
        grid_dst_norm = tgm.transform_points(
            dst_norm_trans_dst_pix, grid_dst_pix)

        # transform norm grid
        dst_norm_M_src_norm = torch.matmul(dst_norm_trans_dst_pix,
            torch.matmul(dst_pix_trans_src_pix, src_pix_trans_src_norm))
        grid_dst_norm2 = tgm.transform_points(
            dst_norm_trans_src_norm, grid_src_norm)

        # grids should be equal
        import ipdb;ipdb.set_trace()
        self.assertTrue(utils.check_equal_torch(
            grid_dst_norm, grid_dst_norm2))

        # warp tensor
        patch = torch.rand(batch_size, 1, src_h, src_w)
        patch_warp = tgm.warp_perspective(patch,
            dst_norm_trans_src_norm, (dst_h, dst_w))
        self.assertTrue(utils.check_equal_torch(
            patch[:, :, :3, 1:3], patch_warped))
        pass

    def test_get_perspective_transform(self):
        # generate input data
        h, w = 64, 32  # height, width
        norm = torch.randn(1, 4, 2)
        points_src = torch.FloatTensor([[
            [0, 0], [h, 0], [0, w], [h, w],
        ]])
        points_dst = points_src + norm

        # compute transform from source to target
        dst_homo_src = tgm.get_perspective_transform(points_src, points_dst)

        res = utils.check_equal_torch(
            tgm.transform_points(dst_homo_src, points_src), points_dst)
        self.assertTrue(res)

    def test_get_perspective_transform_gradcheck(self):
        # generate input data
        h, w = 64, 32  # height, width
        norm = torch.randn(1, 4, 2)
        points_src = torch.FloatTensor([[
            [0, 0], [h, 0], [0, w], [h, w],
        ]])
        points_dst = points_src + norm
        points_src = utils.tensor_to_gradcheck_var(points_src)  # to var
        points_dst = utils.tensor_to_gradcheck_var(points_dst)  # to var

        # compute transform from source to target
        res = gradcheck(tgm.get_perspective_transform,
            (points_src, points_dst,), raise_exception=True)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
