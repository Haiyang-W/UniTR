import torch
from torch import nn
from torch.nn import functional as F
from ...utils import loss_utils

class BEVGridTransform(nn.Module):
    def __init__(
        self,
        input_scope,
        output_scope,
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.classes = class_names
        input_scope = self.model_cfg.GRID_TRANSFORM.INPUT_SCOPE
        output_scope = self.model_cfg.GRID_TRANSFORM.OUTPUT_SCOPE
        self.transform = BEVGridTransform(input_scope, output_scope)
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(True),
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(True),
            nn.Conv2d(input_channels, len(self.classes), 1),
        )
        loss_cfg = self.model_cfg.LOSS_CONFIG
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cfg.gamma,alpha=loss_cfg.alpha)

    def forward(
        self, batch_dict
    ):
        x = batch_dict['spatial_features_2d']
        target = batch_dict['gt_masks_bev']
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        x = self.classifier(x)

        if self.training:
            tb_dict = {}
            loss_all = 0
            for index, name in enumerate(self.classes):
                pred = x[:, index].flatten(1,2).unsqueeze(-1)
                label = target[:, index].flatten(1,2).unsqueeze(-1)
                label_weight = torch.ones_like(label)
                loss = self.loss_cls(pred,label,label_weight).mean()
                tb_dict[f"loss_{name}"] = loss
                loss_all += loss
            batch_dict['loss'] = loss_all
            batch_dict['tb_dict'] = tb_dict
        else:
            batch_dict['masks_bev'] = torch.sigmoid(x)
        return batch_dict
