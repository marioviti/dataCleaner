from protocols import *

class AttrDict(dict):
    """
    AttrDict is a class that can be addressed as a dictionary or struct
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import numpy as np

class TensorItem(np.ndarray):
    def __init__(self, *args, **kwargs):
        super(TensorItem, self).__init__()

    def set_protocol(self, protocol):
        self.protocol=protocol

    def get_protocol(self, protocol):
        return self.protocol

    def is_channel_first(self):
        return is_channel_first(self.protocol)

    def is_channel_last(self):
        return is_channel_last(self.protocol)

    def get_num_dims(self):
        return get_num_dims(self.protocol)
    
    @staticmethod
    def build(protocol,*args,**kwargs):
        # it is not that easy extend numpy.ndarray class, the overide of the
        # constructor is not really pythonic. This workaround will work
        # even for future implementation of numpy.ndarray.
        tensor = TensorItem(*args,**kwargs)
        tensor.set_protocol(protocol)
        return tensor

    @staticmethod
    def fromNumpy(protocol, arrayNumpy):
        tensor = TensorItem(arrayNumpy.shape,arrayNumpy.dtype)
        tensor.set_protocol(protocol) 
        tensor[...] = arrayNumpy[...]
        return tensor

class Batch(AttrDict):
    """
    Extending AttrDict with defined fields xs,ys,ids,extra
    xs: input, ys=f(xs), ids=identifier.
    """
    def __init__(self, xs=None, ys=None, ids=None, extras=None, *args, **kwargs,):
        d = {}
        d['xs'] = xs
        d['ys'] = ys
        d['ids'] = ids
        d['extras'] = extras
        super(Batch, self).__init__(*args, { **kwargs, **d } )