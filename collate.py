import numpy as np
from protocols import CHANNEL_FIRST, CHANNEL_LAST
from batch import TensorItem, Batch

def generic_collate(item_list,channel_order=CHANNEL_LAST):
    """
    Collate single input, single output Item.
    ARGS:
        item_list: (Batch) list.
    """
    xs_list = [np.expand_dims(item.xs,0) for item in item_list]
    ys_list = [np.expand_dims(item.ys,0) for item in item_list]

    ids_list = [item.ids for item in item_list]
    extra_list = [item.extra for item in item_list]
    
    batch_xs = guess_collate(xs_list)(xs_list)
    batch_ys = guess_collate(ys_list)(ys_list)
    batch_ids = ids_list
    batch_extra = extra_list
    
    collacted_batch = Batch(batch_xs, batch_ys, batch_ids, batch_extra)
    return collacted_batch

def guess_collate(array_list):
    """
    Guessing collate from shape.
    """
    l_shape = len(array_list[0].shape)
    assert l_shape > 1 and l_shape < 6
    # N,H,W,D,C or N,C,H,W,D
    if l_shape == 5:
        return array_3D_collate
    # N,H,W,C or N,C,H,W
    if l_shape == 4:
        return array_2D_collate
    # N,L,C or N,C,L
    if l_shape == 3:
        return array_1D_collate    

def array_1D_collate(array_list):
    """
    Collate 1D volumes of shape l_i,c into a batch volume
    of size n,l*,c where d* = max(d_i) and n = len(array_list).
    """
    arrs = array_list[0]
    for arr in array_list[1:]: 
        arrs = np.concatenate([arrs,arr],axis=0)
    return arrs

def array_2D_collate(array_list, channel_last=True):
    """
    Collate 2D volumes of shape w_i,h_i,c into a batch volume
    of size n,w*,h*,c where d* = max(d_i) and n = len(array_list).
    """
    i_slice = slice(1:3) if channel_last else slice(2:)
    i_idx = -1 if channel_last else 1

    batch_num = len(array_list)
    if batch_num == 1:
        return array_list[0]
    
    # getting containing shape
    containing_shape = [0,0]
    for i,array in enumerate(array_list):
        curr_shape = array.shape[i_slice]
        containing_shape = [np.max([a,b]) for a,b in \
                            zip(containing_shape,curr_shape) ]
    # Initialize batch
    channels = array_list[0].shape[i_idx]
    batch = np.zeros([batch_num] + containing_shape + [channels])
    biggest_shape = batch.shape
    
    for i,array in enumerate(array_list):
        # centering image to containig volume
        smaller_shape = array.shape
        sh,sw = smaller_shape[i_slice]
        bh,bw = biggest_shape[i_slice]
        dh,dw = bh-sh, bw-sw
        q2dh,r2dh,q2dw,r2dw = dh//2,dh%2,dw//2,dw%2

        if channel_last:
            batch[
                i,
                q2dh:bh-(q2dh+r2dh),
                q2dw:bw-(q2dw+r2dw),:
                ] = array[0,...]
        else:
            batch[
                i,:,
                q2dh:bh-(q2dh+r2dh),
                q2dw:bw-(q2dw+r2dw)
                ] = array[0,...]
    
    return batch 

def array_3D_collate(array_list, channel_last=True):
    """
    Collate 3D volumes of shape d_i,w_i,h_i,c into a batch volume
    of size n,d*,w*,h*,c where d* = max(d_i) and n = len(array_list).
    """
    batch_num = len(array_list)
    if batch_num == 1:
        return array_list[0]
    
    # getting containing (biggest) shape
    containing_shape = [0,0,0]
    i_slice = slice(1:4) if channel_last else slice(2:)
    i_idx = -1 if channel_last else 1
    for i,array in enumerate(array_list):
        curr_shape = array.shape[i_slice]
        containing_shape = [ np.max([a,b]) for a,b in \
                             zip(containing_shape,curr_shape) ]
    # Initialize batch
    channels = array_list[0].shape[i_idx]
    batch = np.zeros([batch_num]+containing_shape+[channels])
    biggest_shape = batch.shape
    
    for i,array in enumerate(array_list):        
        # centering image to containig volume
        smaller_shape = array.shape
        sh,sw,sd = smaller_shape[i_slice] 
        bh,bw,bd = biggest_shape[i_slice]
        dh,dw,dd = bh-sh, bw-sw, bd-sd
        q2dh,r2dh,q2dw,r2dw,q2dd,r2dd = dh//2,dh%2,dw//2,dw%2,dd//2,dd%2

        if channel_last:
            batch[
                i,
                q2dh:bh-(q2dh+r2dh),
                q2dw:bw-(q2dw+r2dw),
                q2dd:bd-(q2dd+r2dd),:
                ] = array[0,...]
        else:
            batch[
                i,:,
                q2dh:bh-(q2dh+r2dh),
                q2dw:bw-(q2dw+r2dw),
                q2dd:bd-(q2dd+r2dd)
                ] = array[0,...]
    return batch 