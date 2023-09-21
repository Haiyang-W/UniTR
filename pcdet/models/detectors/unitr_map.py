from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms, mm_backbone,dense_heads
from ..backbones_image import img_neck
from ..backbones_2d import fuser
from ...utils.spconv_utils import find_all_spconv_keys

class UniTRMAP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe','mm_backbone', 'map_to_bev_module',
            'neck','vtransform', 'fuser',
            'backbone_2d','dense_head',
        ]
        self.module_list = self.build_networks()
        self.time_list = []
       
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def build_mm_backbone(self, model_info_dict):
        if self.model_cfg.get('MM_BACKBONE', None) is None:
            return None, model_info_dict
        mm_backbone_name = self.model_cfg.MM_BACKBONE.NAME
        del self.model_cfg.MM_BACKBONE['NAME']
        mm_backbone_module = mm_backbone.__all__[mm_backbone_name](
            model_cfg=self.model_cfg.MM_BACKBONE,
            use_map=self.dataset.use_map
            )
        model_info_dict['module_list'].append(mm_backbone_module)

        return mm_backbone_module, model_info_dict
    
    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=len(self.dataset.map_classes) if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.dataset.map_classes,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # adapt pretrain image backbone to mm backbone
            if 'image_backbone' in key:
                key = key.replace("image","mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer","image_input_layer")


            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("not exist",key)
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            ret_dict = [{
                'masks_bev': masks_bev,
                'gt_masks_bev': gt_masks_bev
            } for masks_bev,gt_masks_bev in zip(batch_dict['masks_bev'],batch_dict['gt_masks_bev'])]
            return ret_dict

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict
