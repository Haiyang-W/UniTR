import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool

__all__ = ["LSSTransform"]

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class LSSTransform(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE
        self.accelerate = self.model_cfg.get("ACCELERATE",False)
        if self.accelerate:
            self.cache = None

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channel,
                    out_channel,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            # map segmentation
            if self.model_cfg.get('USE_CONV_FOR_NO_STRIDE',False):
                self.downsample = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                )
            else:
                self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        x = x.to(torch.float)
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        lidar2img,
        img_aug_matrix
    ):
        lidar2img = lidar2img.to(torch.float)
        img_aug_matrix = img_aug_matrix.to(torch.float)

        B,N = lidar2img.shape[:2]
        D,H,W = self.frustum.shape[:3]
        points = self.frustum.view(1,1,D,H,W,3).repeat(B,N,1,1,1,1)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = torch.cat([points,torch.ones_like(points[...,-1:])],dim=-1)
        points = torch.inverse(img_aug_matrix).view(B,N,1,1,1,4,4).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
                torch.ones_like(points[:, :, :, :, :, 2:3])
            ),
            5,
        )
        points = torch.inverse(lidar2img).view(B,N,1,1,1,4,4).matmul(points).squeeze(-1)[...,:3]

        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float)
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        if self.accelerate and self.cache is None:
            self.cache = (geom_feats,kept)

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def acc_bev_pool(self,x):
        geom_feats,kept = self.cache
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)
        x = x[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(self, batch_dict):
        x = batch_dict['image_fpn'] 
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        img = x.view(int(BN/6), 6, C, H, W)
        x = self.get_cam_feats(img)
        if self.accelerate and self.cache is not None:
            x = self.acc_bev_pool(x)
        else:
            img_aug_matrix = batch_dict['img_aug_matrix']
            lidar2image = batch_dict['lidar2image']
            if self.training and 'lidar2image_aug' in batch_dict:
                lidar2image = batch_dict['lidar2image_aug']
             
            geom = self.get_geometry(
                lidar2image,
                img_aug_matrix
            )
            x = self.bev_pool(geom, x)
        x = self.downsample(x)
        batch_dict['spatial_features_img'] = x.permute(0,1,3,2).contiguous()
        return batch_dict
