import torch
from pdb import set_trace
import random
import numpy as np


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    '''
    Args:
        batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
        voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
    Returns:
        flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
            Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
    '''
    device = batch_win_inds.device

    flat2window_inds_dict = {}

    for dl in drop_info: # dl: short for drop level

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])

        num_windows = len(torch.unique(conti_win_inds))
        max_tokens = drop_info[dl]['max_tokens']

        inner_win_inds = get_inner_win_inds(conti_win_inds)

        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

        flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

        if debug:
            assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
            assert (flat2window_inds >= 0).all()
            max_ind = flat2window_inds.max().item()
            assert  max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
            assert  max_ind >= (num_windows-1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})'

    return flat2window_inds_dict


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info):
    '''
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    
    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    '''
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        feat_3d = torch.zeros((num_windows * max_tokens, feat_dim), dtype=dtype, device=device)
        if this_inds.max() >= num_windows * max_tokens:
            set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict

def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    assert (check_feat == 0).all()
    
    return all_flat_feat

def get_flat2win_inds_v2(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug)
    # add voxel_drop_lvl and batching_info into transform_dict for better wrapping
    transform_dict['voxel_drop_level'] = voxel_drop_lvl
    transform_dict['batching_info'] = drop_info
    return transform_dict
    
def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)

def flat2window_v2(feat, inds_dict):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info)


@torch.no_grad()
def get_inner_win_inds(win_inds):
    '''
    Args:
        win_inds indicates which windows a voxel belongs to. Voxels share a window have same inds.
        shape = [N,]
    Return:
        inner_inds: shape=[N,]. Indicates voxel's id in a window. if M voxels share a window, their inner_inds would
            be torch.arange(m, dtype=torch.long)
    Note that this function might output different results from get_inner_win_inds_slow due to the unstable pytorch sort.
    '''

    sort_inds, order = win_inds.sort(stable=True) # sort_inds is like [0,1, 2, 0, 0, 1] -> [0,0,0, 1, 2,2]
    roll_inds_left = torch.roll(sort_inds, -1) # [0,0, 1, 2,2,0]

    diff = sort_inds - roll_inds_left # [0, 0, -1, -1, 0, 2]
    end_pos_mask = diff != 0

    bincount = torch.bincount(win_inds)
    # assert bincount.max() <= max_tokens
    unique_sort_inds, _ = torch.sort(torch.unique(win_inds))
    num_tokens_each_win = bincount[unique_sort_inds] #[3, 1, 2]

    template = torch.ones_like(win_inds) #[1,1,1, 1, 1,1]
    template[end_pos_mask] = (num_tokens_each_win-1) * -1 #[1,1,-2, 0, 1,-1]

    inner_inds = torch.cumsum(template, 0) #[1,2,0, 0, 1,0]
    inner_inds[end_pos_mask] = num_tokens_each_win #[1,2,3, 1, 1,2]
    inner_inds -= 1 #[0,1,2, 0, 0,1]


    #recover the order
    inner_inds_reorder = -torch.ones_like(win_inds)
    inner_inds_reorder[order] = inner_inds

    ##sanity check
    assert (inner_inds >= 0).all()
    assert (inner_inds == 0).sum() == len(unique_sort_inds)
    assert (num_tokens_each_win > 0).all()
    random_win = unique_sort_inds[random.randint(0, len(unique_sort_inds)-1)]
    random_mask = win_inds == random_win
    num_voxel_this_win = bincount[random_win].item()
    random_inner_inds = inner_inds_reorder[random_mask] 

    assert len(torch.unique(random_inner_inds)) == num_voxel_this_win
    assert random_inner_inds.max() == num_voxel_this_win - 1
    assert random_inner_inds.min() == 0

    return inner_inds_reorder

@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift, shift_list=None, return_win_coors=False):

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z
    
    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z

    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z

    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
    if return_win_coors:
        batch_win_coords = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)
        return batch_win_inds, coors_in_win, batch_win_coords
    
    return batch_win_inds, coors_in_win

@torch.no_grad()
def make_continuous_inds(inds):

    ### make batch_win_inds continuous
    dtype = inds.dtype
    device = inds.device

    unique_inds, _ = torch.sort(torch.unique(inds))
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    canvas = -torch.ones((max_origin_inds+1,), dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

    conti_inds = canvas[inds]

    assert conti_inds.max() == len(torch.unique(conti_inds)) - 1, 'Continuity check failed.'
    assert conti_inds.min() == 0, '-1 in canvas should not be indexed.'
    return conti_inds

