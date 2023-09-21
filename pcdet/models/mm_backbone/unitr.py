import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pcdet.models.model_utils.swin_utils import PatchEmbed
from pcdet.models.model_utils.unitr_utils import MapImage2Lidar, MapLidar2Image
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned
from pcdet.models.backbones_3d.dsvt import _get_activation_fn, DSVTInputLayer
from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds
get_inner_win_inds_cuda = ingroup_inds

class UniTR(nn.Module):
    '''
    UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation.
    Main args:
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        checkpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
        accelerate (bool): whether accelerate forward by caching image pos embed, image2lidar coords and lidar2image coords.
    '''

    def __init__(self, model_cfg, use_map=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.set_info = set_info = self.model_cfg.set_info
        self.d_model = d_model = self.model_cfg.d_model
        self.nhead = nhead = self.model_cfg.nhead
        self.stage_num = stage_num = 1  # only support plain bakbone
        self.num_shifts = [2] * self.stage_num
        self.checkpoint_blocks = self.model_cfg.checkpoint_blocks
        self.image_pos_num, self.lidar_pos_num = set_info[0][-1], set_info[0][-1]
        self.accelerate = self.model_cfg.get('ACCELERATE', False)
        self.use_map = use_map

        self.image_input_layer = UniTRInputLayer(
            self.model_cfg.IMAGE_INPUT_LAYER, self.accelerate)
        self.lidar_input_layer = UniTRInputLayer(
            self.model_cfg.LIDAR_INPUT_LAYER)

        # image patch embedding
        patch_embed_cfg = self.model_cfg.PATCH_EMBED
        self.patch_embed = PatchEmbed(
            in_channels=patch_embed_cfg.in_channels,
            embed_dims=patch_embed_cfg.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_embed_cfg.patch_size,
            stride=patch_embed_cfg.patch_size,
            norm_cfg=patch_embed_cfg.norm_cfg if patch_embed_cfg.patch_norm else None
        )
        patch_size = [patch_embed_cfg.image_size[0] // patch_embed_cfg.patch_size,
                      patch_embed_cfg.image_size[1] // patch_embed_cfg.patch_size]
        self.patch_size = patch_size
        patch_x, patch_y = torch.meshgrid(torch.arange(
            patch_size[0]), torch.arange(patch_size[1]))
        patch_z = torch.zeros((patch_size[0] * patch_size[1], 1))
        self.patch_zyx = torch.cat(
            [patch_z, patch_y.reshape(-1, 1), patch_x.reshape(-1, 1)], dim=-1).cuda()
        # patch coords with batch id
        self.patch_coords = None

        # image branch output norm
        self.out_indices = self.model_cfg.out_indices
        for i in self.out_indices:
            layer = nn.LayerNorm(d_model[-1])
            layer_name = f'out_norm{i}'
            self.add_module(layer_name, layer)

        # Sparse Regional Attention Blocks
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        layer_cfg = self.model_cfg.layer_cfg
        block_id = 0
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_list, norm_list = [], []
            for i in range(num_blocks_this_stage):
                block_list.append(
                    UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                               dropout, activation, batch_first=True, block_id=block_id,
                               dout=dmodel_this_stage, layer_cfg=layer_cfg)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
                block_id += 1
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(
                f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))
            if layer_cfg.get('split_residual', False):
                # use different norm for lidar and image
                lidar_norm_list = [nn.LayerNorm(
                    dmodel_this_stage) for _ in range(num_blocks_this_stage)]
                self.__setattr__(
                    f'lidar_residual_norm_stage_{stage_id}', nn.ModuleList(lidar_norm_list))

        # Fuse Backbone
        fuse_cfg = self.model_cfg.get('FUSE_BACKBONE', None)
        self.fuse_on = fuse_cfg is not None
        if self.fuse_on:
            # image2lidar
            image2lidar_cfg = fuse_cfg.get('IMAGE2LIDAR', None)
            self.image2lidar_on = image2lidar_cfg is not None
            if self.image2lidar_on:
                # block range of image2lidar
                self.image2lidar_start = image2lidar_cfg.block_start
                self.image2lidar_end = image2lidar_cfg.block_end
                self.map_image2lidar_layer = MapImage2Lidar(
                    image2lidar_cfg, self.accelerate, self.use_map)
                self.image2lidar_input_layer = UniTRInputLayer(
                    image2lidar_cfg.image2lidar_layer)
                self.image2lidar_pos_num = image2lidar_cfg.image2lidar_layer.set_info[0][1]
                # encode the position of each patch from the closest point in image space
                self.neighbor_pos_embed = PositionEmbeddingLearned(
                    2, self.d_model[-1])

            # lidar2image
            lidar2image_cfg = fuse_cfg.get('LIDAR2IMAGE', None)
            self.lidar2image_on = lidar2image_cfg is not None
            if self.lidar2image_on:
                # block range of lidar2image
                self.lidar2image_start = lidar2image_cfg.block_start
                self.lidar2image_end = lidar2image_cfg.block_end
                self.map_lidar2image_layer = MapLidar2Image(
                    lidar2image_cfg, self.accelerate, self.use_map)
                self.lidar2image_input_layer = UniTRInputLayer(
                    lidar2image_cfg.lidar2image_layer)
                self.lidar2image_pos_num = lidar2image_cfg.lidar2image_layer.set_info[0][1]

        self._reset_parameters()

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - camera_imgs (Tensor[float]): multi view images, shape of (B, N, C, H, W),
                    where N is the number of image views.
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - image_features (Tensor[float]):
        '''
        # lidar(3d) and image(2d) preprocess
        multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list = self._input_preprocess(
            batch_dict)
        # lidar(3d) and image(3d) preprocess
        if self.image2lidar_on:
            image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list = self._image2lidar_preprocess(
                batch_dict, multi_feat, multi_pos_embed_list)
        # lidar(2d) and image(2d) preprocess
        if self.lidar2image_on:
            lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list = self._lidar2image_preprocess(
                batch_dict, multi_feat, multi_pos_embed_list)
        output = multi_feat
        block_id = 0
        voxel_num = batch_dict['voxel_num']
        batch_dict['image_features'] = []
        # block forward
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(
                f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()

                if self.image2lidar_on and i >= self.image2lidar_start and i < self.image2lidar_end:
                    output = block(output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                elif self.lidar2image_on and i >= self.lidar2image_start and i < self.lidar2image_end:
                    output = block(output, lidar2image_inds_list[stage_id], lidar2image_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                else:
                    output = block(output, multi_set_voxel_inds_list[stage_id], multi_set_voxel_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                   block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                # use different norm for lidar and image
                if self.model_cfg.layer_cfg.get('split_residual', False):
                    output = torch.cat([self.__getattr__(f'lidar_residual_norm_stage_{stage_id}')[i](output[:voxel_num] + residual[:voxel_num]),
                                       residual_norm_layers[i](output[voxel_num:] + residual[voxel_num:])], dim=0)
                else:
                    output = residual_norm_layers[i](output + residual)
                block_id += 1
                # recover image feature shape
                if i in self.out_indices:
                    batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                 coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                    batch_dict['image_features'].append(batch_spatial_features)
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output[:voxel_num]
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict

    def _input_preprocess(self, batch_dict):
        # image branch
        imgs = batch_dict['camera_imgs']
        B, N, C, H, W = imgs.shape  # 6, 6, 3, 256, 704
        imgs = imgs.view(B * N, C, H, W)

        imgs, hw_shape = self.patch_embed(imgs)  # 8x [36, 2816, C] [32, 88]
        batch_dict['hw_shape'] = hw_shape

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1])
        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone()[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
        patch_info = self.image_input_layer(batch_dict)
        patch_feat = batch_dict['patch_features']
        patch_set_voxel_inds_list = [[patch_info[f'set_voxel_inds_stage{s}_shift{i}']
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_set_voxel_masks_list = [[patch_info[f'set_voxel_mask_stage{s}_shift{i}']
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_pos_embed_list = [[[patch_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                  for i in range(self.num_shifts[s])] for b in range(self.image_pos_num)] for s in range(len(self.set_info))]

        # lidar branch
        voxel_info = self.lidar_input_layer(batch_dict)
        voxel_feat = batch_dict['voxel_features']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
                                for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}']
                                 for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                            for i in range(self.num_shifts[s])] for b in range(self.lidar_pos_num)] for s in range(len(self.set_info))]

        # multi-modality parallel
        voxel_num = voxel_feat.shape[0]
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0)
        multi_set_voxel_inds_list = [[torch.cat([set_voxel_inds_list[s][i], patch_set_voxel_inds_list[s][i]+voxel_num], dim=1)
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        multi_set_voxel_masks_list = [[torch.cat([set_voxel_masks_list[s][i], patch_set_voxel_masks_list[s][i]], dim=1)
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        multi_pos_embed_list = []
        for s in range(len(self.set_info)):
            block_pos_embed_list = []
            for b in range(self.set_info[s][1]):
                shift_pos_embed_list = []
                for i in range(self.num_shifts[s]):
                    if b < self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            torch.cat([pos_embed_list[s][b][i], patch_pos_embed_list[s][b][i]], dim=0))
                    elif b < self.lidar_pos_num and b >= self.image_pos_num:
                        shift_pos_embed_list.append(pos_embed_list[s][b][i])
                    elif b >= self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            patch_pos_embed_list[s][b][i])
                    else:
                        raise NotImplementedError
                block_pos_embed_list.append(shift_pos_embed_list)
            multi_pos_embed_list.append(block_pos_embed_list)

        return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list

    def _image2lidar_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        N = batch_dict['camera_imgs'].shape[1]
        voxel_num = batch_dict['voxel_num']
        image2lidar_coords_zyx, nearest_dist = self.map_image2lidar_layer(
            batch_dict)
        image2lidar_coords_bzyx = torch.cat(
            [batch_dict['patch_coords'][:, :1].clone(), image2lidar_coords_zyx], dim=1)
        image2lidar_coords_bzyx[:, 0] = image2lidar_coords_bzyx[:, 0] // N
        image2lidar_batch_dict = {}
        image2lidar_batch_dict['voxel_features'] = multi_feat.clone()
        image2lidar_batch_dict['voxel_coords'] = torch.cat(
            [batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)
        image2lidar_info = self.image2lidar_input_layer(image2lidar_batch_dict)
        image2lidar_inds_list = [[image2lidar_info[f'set_voxel_inds_stage{s}_shift{i}']
                                  for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        image2lidar_masks_list = [[image2lidar_info[f'set_voxel_mask_stage{s}_shift{i}']
                                   for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        image2lidar_pos_embed_list = [[[image2lidar_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                        for i in range(self.num_shifts[s])] for b in range(self.image2lidar_pos_num)] for s in range(len(self.set_info))]
        image2lidar_neighbor_pos_embed = self.neighbor_pos_embed(nearest_dist)

        for b in range(self.image2lidar_start, self.image2lidar_end):
            for i in range(self.num_shifts[0]):
                image2lidar_pos_embed_list[0][b -
                                              self.image2lidar_start][i][voxel_num:] += image2lidar_neighbor_pos_embed
                multi_pos_embed_list[0][b][i] += image2lidar_pos_embed_list[0][b -
                                                                               self.image2lidar_start][i]
        return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list

    def _lidar2image_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        N = batch_dict['camera_imgs'].shape[1]
        hw_shape = batch_dict['hw_shape']
        lidar2image_coords_zyx = self.map_lidar2image_layer(batch_dict)
        lidar2image_coords_bzyx = torch.cat(
            [batch_dict['voxel_coords'][:, :1].clone(), lidar2image_coords_zyx], dim=1)
        multiview_coords = batch_dict['patch_coords'].clone()
        multiview_coords[:, 0] = batch_dict['patch_coords'][:, 0] // N
        multiview_coords[:, 1] = batch_dict['patch_coords'][:, 0] % N
        multiview_coords[:, 2] += hw_shape[1]
        multiview_coords[:, 3] += hw_shape[0]
        lidar2image_batch_dict = {}
        lidar2image_batch_dict['voxel_features'] = multi_feat.clone()
        lidar2image_batch_dict['voxel_coords'] = torch.cat(
            [lidar2image_coords_bzyx, multiview_coords], dim=0)
        lidar2image_info = self.lidar2image_input_layer(lidar2image_batch_dict)
        lidar2image_inds_list = [[lidar2image_info[f'set_voxel_inds_stage{s}_shift{i}']
                                  for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        lidar2image_masks_list = [[lidar2image_info[f'set_voxel_mask_stage{s}_shift{i}']
                                   for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        lidar2image_pos_embed_list = [[[lidar2image_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                        for i in range(self.num_shifts[s])] for b in range(self.lidar2image_pos_num)] for s in range(len(self.set_info))]

        for b in range(self.lidar2image_start, self.lidar2image_end):
            for i in range(self.num_shifts[0]):
                multi_pos_embed_list[0][b][i] += lidar2image_pos_embed_list[0][b -
                                                                               self.lidar2image_start][i]
        return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _recover_image(self, pillar_features, coords, indices):
        pillar_features = getattr(self, f'out_norm{indices}')(pillar_features)
        batch_size = coords[:, 0].max().int().item() + 1
        batch_spatial_features = pillar_features.view(
            batch_size, self.patch_size[0], self.patch_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return batch_spatial_features

class UniTRBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, block_id=-100, dout=None, layer_cfg=dict()):
        super().__init__()

        encoder_1 = UniTR_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       activation, batch_first, layer_cfg=layer_cfg)
        encoder_2 = UniTR_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       activation, batch_first, dout=dout, layer_cfg=layer_cfg)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
            voxel_num=0,
            using_checkpoint=False,
    ):
        num_shifts = len(set_voxel_inds_list)

        output = src
        for i in range(num_shifts):
            set_id = block_id % 2
            shift_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[shift_id]
            layer = self.encoder_list[i]
            if using_checkpoint and self.training:
                output = checkpoint(
                    layer, output, set_voxel_inds, set_voxel_masks, pos_embed, voxel_num)
            else:
                output = layer(output, set_voxel_inds,
                               set_voxel_masks, pos_embed, voxel_num=voxel_num)

        return output

class UniTR_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0, dout=None, layer_cfg=dict()):
        super().__init__()
        self.win_attn = SetAttention(
            d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout, layer_cfg)
        if dout is None:
            dout = d_model
        self.norm = nn.LayerNorm(dout)
        self.d_model = d_model

    def forward(self, src, set_voxel_inds, set_voxel_masks, pos=None, voxel_num=0):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks,
                            set_voxel_inds, voxel_num=voxel_num)
        src = src + identity
        src = self.norm(src)

        return src
class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead
        if batch_first:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.layer_cfg = layer_cfg

        use_bn = layer_cfg.get('use_bn', False)
        if use_bn:
            assert use_bn is False
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        if layer_cfg.get('split_ffn', False):
            # Implementation of lidar Feedforward model
            self.lidar_linear1 = nn.Linear(d_model, dim_feedforward)
            self.lidar_dropout = nn.Dropout(mlp_dropout)
            self.lidar_linear2 = nn.Linear(dim_feedforward, d_model)

            use_bn = layer_cfg.get('use_bn', False)
            if use_bn:
                assert use_bn is False
            else:
                self.lidar_norm1 = nn.LayerNorm(d_model)
                self.lidar_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None, voxel_num=0):
        set_features = src[voxel_inds]  # [win_num, 36, d_model]
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features
        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]

        flatten_inds = voxel_inds.reshape(-1)
        unique_flatten_inds, inverse = torch.unique(
            flatten_inds, return_inverse=True)
        perm = torch.arange(inverse.size(
            0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(
            unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
        src2 = src2.reshape(-1, self.d_model)[perm]

        if self.layer_cfg.get('split_ffn', False):
            src = src + self.dropout1(src2)
            lidar_norm = self.lidar_norm1(src[:voxel_num])
            image_norm = self.norm1(src[voxel_num:])
            src = torch.cat([lidar_norm, image_norm], dim=0)

            lidar_linear2 = self.lidar_linear2(self.lidar_dropout(
                self.activation(self.lidar_linear1(src[:voxel_num]))))
            image_linear2 = self.linear2(self.dropout(
                self.activation(self.linear1(src[voxel_num:]))))
            src2 = torch.cat([lidar_linear2, image_linear2], dim=0)

            src = src + self.dropout2(src2)
            lidar_norm2 = self.lidar_norm2(src[:voxel_num])
            image_norm2 = self.norm2(src[voxel_num:])
            src = torch.cat([lidar_norm2, image_norm2], dim=0)
        else:
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(
                self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class UniTRInputLayer(DSVTInputLayer):
    ''' 
    This class converts the output of vfe to unitr input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        input_image (bool): whether input modal is image.
    '''

    def __init__(self, model_cfg, accelerate=False):
        # dummy config
        model_cfg.downsample_stride = model_cfg.get('downsample_stride',[])
        model_cfg.normalize_pos = model_cfg.get('normalize_pos',False)
        super().__init__(model_cfg)

        self.input_image = self.model_cfg.get('input_image', False)
        self.key_name = 'patch' if self.input_image else 'voxel'
        # only support image input accelerate
        self.accelerate = self.input_image and accelerate
        self.process_info = None

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...

        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - ...
        '''
        if self.input_image and self.process_info is not None and (batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]):
            patch_info = dict()
            for k in (self.process_info.keys()):
                if torch.is_tensor(self.process_info[k]):
                    patch_info[k] = self.process_info[k].clone()
                else:
                    patch_info[k] = copy.deepcopy(self.process_info[k])
            # accelerate by caching pos embed as patch coords are fixed
            if not self.accelerate:
                for stage_id in range(len(self.downsample_stride)+1):
                    for block_id in range(self.set_info[stage_id][1]):
                        for shift_id in range(self.num_shifts[stage_id]):
                            patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                                self.get_pos_embed(
                                    patch_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
            return patch_info

        key_name = self.key_name
        coors = batch_dict[f'{key_name}_coords'].long()

        info = {}
        # original input voxel coors
        info[f'voxel_coors_stage0'] = coors.clone()

        for stage_id in range(len(self.downsample_stride)+1):
            # window partition of corrsponding stage-map
            info = self.window_partition(info, stage_id)
            # generate set id of corrsponding stage-map
            info = self.get_set(info, stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                        self.get_pos_embed(
                            info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)

        info['sparse_shape_list'] = self.sparse_shape_list

        # save process info for image input as patch coords are fixed
        if self.input_image:
            self.process_info = {}
            for k in (info.keys()):
                if k != 'patch_feats_stage0':
                    if torch.is_tensor(info[k]):
                        self.process_info[k] = info[k].clone()
                    else:
                        self.process_info[k] = copy.deepcopy(info[k])

        return info
    
    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        '''
        voxel_order_list[list]: order respectively sort by x, y, z
        '''

        device = batch_win_inds.device

        # max number of voxel in a window
        voxel_num_set = self.set_info[stage_id][0]
        max_voxel = self.window_shape[stage_id][shift_id][0] * \
            self.window_shape[stage_id][shift_id][1] * \
            self.window_shape[stage_id][shift_id][2]

        if self.model_cfg.get('expand_max_voxels', None) is not None:
            max_voxel *= self.model_cfg.get('expand_max_voxels', None)
        contiguous_win_inds = torch.unique(
            batch_win_inds, return_inverse=True)[1]
        voxelnum_per_win = torch.bincount(contiguous_win_inds)
        win_num = voxelnum_per_win.shape[0]

        setnum_per_win_float = voxelnum_per_win / voxel_num_set
        setnum_per_win = torch.ceil(setnum_per_win_float).long()

        set_num = setnum_per_win.sum().item()
        setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1]

        set_win_inds = torch.full((set_num,), 0, device=device)
        set_win_inds[setnum_per_win_cumsum] = 1
        set_win_inds = torch.cumsum(set_win_inds, dim=0)

        # input [0,0,0, 1, 2,2]
        roll_set_win_inds_left = torch.roll(
            set_win_inds, -1)  # [0,0, 1, 2,2,0]
        diff = set_win_inds - roll_set_win_inds_left  # [0, 0, -1, -1, 0, 2]
        end_pos_mask = diff != 0
        template = torch.ones_like(set_win_inds)
        template[end_pos_mask] = (setnum_per_win - 1) * -1  # [1,1,-2, 0, 1,-1]
        set_inds_in_win = torch.cumsum(template, dim=0)  # [1,2,0, 0, 1,0]
        set_inds_in_win[end_pos_mask] = setnum_per_win  # [1,2,3, 1, 1,2]
        set_inds_in_win = set_inds_in_win - 1  # [0,1,2, 0, 0,1]

        offset_idx = set_inds_in_win[:, None].repeat(
            1, voxel_num_set) * voxel_num_set
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * \
            voxelnum_per_win[set_win_inds][:, None]
        base_select_idx = base_select_idx.double(
        ) / (setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        base_select_idx = torch.floor(base_select_idx)

        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel

        # sort by y
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
        _, order1 = torch.sort(global_voxel_inds)
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)

        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1])
        inner_voxel_inds_sorty_reorder = inner_voxel_inds_sorty
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty_reorder + \
            max_voxel * contiguous_win_inds
        voxel_inds_padding_sorty = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(
            0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device)

        # sort by x
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)

        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1])
        inner_voxel_inds_sortx_reorder = inner_voxel_inds_sortx
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx_reorder + \
            max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(
            0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device)

        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]
        all_set_voxel_inds = torch.stack(
            (set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)

        return all_set_voxel_inds

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            if self.sparse_shape[-1] == 1:
                ndim = 2
            else:
                ndim = 3
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
