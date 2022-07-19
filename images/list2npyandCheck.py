import numpy as np

import numpy as np
from numpy import dot
from numpy.linalg import norm

def get_cosine_value(LOGGER, obj1_sig, obj2_sig) -> float:
    """get_cosine_value
    Args:
        obj1_sig (list): obj1 signature
        obj2_sig (list): obj2 signature
    Returns:
        float: result
    """
    result = 0.0
    try:
        if 'int' not in type(obj1_sig).__name__ and 'int' not in type(obj2_sig).__name__:
            result = dot(obj1_sig, obj2_sig)/(norm(obj1_sig)*norm(obj2_sig))
    except Exception as ex:
        print("caught exception: ", ex)
        pass
    return result

f = open('/app/images/vsig_pth.txt', 'r')
listData = f.read()
pthData = np.asarray(listData)
f.close()
print("pth_shape: ", pthData.shape)


trtdata = np.load('/app/images/vsig_trt.npy')
trtdata = trtdata.tolist()[0]
trtdata = np.asarray(trtdata)
print("trt_shape: ",trtdata.shape)
#print(npydata.tolist()[0])

retValue = get_cosine_value(None, pthData, trtdata)
print("retValue: ", retValue)






