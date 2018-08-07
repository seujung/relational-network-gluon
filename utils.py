import mxnet as mx
import mxnet.ndarray as F
import numpy as np 


def display_image(data):
    img = data[0]
    img = np.swapaxes(img,0,2)
    plt.imshow((img * 255.0).astype('uint8'))
    
# prepare coord tensor
def cvt_coord(i):
    return [(i/5-2)/2., (i%5-2)/2.]

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

def ndarray_conv(data, i,bs):
    img = F.array(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = F.array(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = F.array(np.asarray(data[2][bs*i:bs*(i+1)]))

    return img, qst, ans