from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms, mm_backbone
from ..backbones_image import img_neck
from ..backbones_2d import fuser
from ...utils.spconv_utils import find_all_spconv_keys

class UniTR(Detector3DTemplate):
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
            model_cfg=self.model_cfg.MM_BACKBONE
            )
        model_info_dict['module_list'].append(mm_backbone_module)

        return mm_backbone_module, model_info_dict
    
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
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
