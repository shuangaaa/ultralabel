import numpy as np
from pycocotools.mask import encode


def mask_to_rle(mask):
    result = encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    return {"size": list(result["size"]), "counts": result["counts"].decode("utf-8"), "order": "F"}
