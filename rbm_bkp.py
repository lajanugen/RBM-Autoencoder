import theano
import numpy as np
import theano.tensor as T
import os
import gzip
import cPickle
from utils import tile_raster_images
import Image
from theano.tensor.shared_randomstreams import RandomStreams

dataset = '../data/mnist.pkl.gz'
datafile = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(datafile)
datafile.close()

lr = 0.1
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;
MB_size = 100
momentum = initialmomentum

numpy_rng = np.random.RandomState(1234)
theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

n_visible = 784
n_hidden = 1000
N = train_set[0].shape[0]
d = train_set[0].shape[1]

data_x = train_set[0]
#print data_x[0]

np.random.shuffle(data_x)
data_x_shared = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))

#v = theano.shared(np.asarray(np.random.randn(n_visible),dtype=theano.config.floatX))
#h = theano.shared(np.asarray(np.random.randn(n_hidden),dtype=theano.config.floatX))
v = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX))
h = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX))

#initial_W = np.asarray(
#    numpy_rng.uniform(
#        low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
#        high=4 * np.sqrt(6. / (n_hidden + n_visible)),
#        size=(n_visible, n_hidden)
#    ),
#    dtype=theano.config.floatX
#)
initial_W = 0.1*np.asarray(
    np.random.randn(n_visible, n_hidden),
    dtype=theano.config.floatX
)
w = theano.shared(value=initial_W,borrow=True)

x = T.matrix()

def get_upds(inp):
    w_update = T.dot(T.transpose(x),T.nnet.sigmoid(T.dot(inp,w)+h))*(1.0/MB_size)
    h_update = T.mean(T.nnet.sigmoid(T.dot(inp,w)+h),axis=0)
    v_update = T.mean(inp,axis=0)
    return w_update,h_update,v_update

neg_sample = theano.shared(value=data_x[0:MB_size])
#neg_samples = theano.shared(value=np.zeros((N,d),dtype=theano.config.floatX))

#h_given_v_p = T.nnet.sigmoid(T.dot(neg_sample,w)+h)
h_given_v_p = T.nnet.sigmoid(T.dot(x,w)+h)
h_given_v_sample = theano_rng.binomial(n=1,p=h_given_v_p,dtype=theano.config.floatX)
v_given_h_p = T.nnet.sigmoid(T.dot(h_given_v_sample,T.transpose(w))+v)
v_given_h_sample = theano_rng.binomial(n=1,p=v_given_h_p,dtype=theano.config.floatX)

pos_w, pos_h, pos_v = get_upds(x)
neg_w, neg_h, neg_v = get_upds(v_given_h_p)

w_upd = theano.shared(value=np.zeros((n_visible,n_hidden),dtype=theano.config.floatX))
v_upd = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX))
h_upd = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX))

w_upd_ = momentum*w_upd + lr*((pos_w - neg_w) - weightcost*w)
v_upd_ = momentum*v_upd + lr*(pos_v - neg_v)
h_upd_ = momentum*h_upd + lr*(pos_h - neg_h)

index = T.lscalar()  # index to a [mini]batch

update = [(w, w + w_upd_),(h, h + h_upd_),(v, v + v_upd_),(w_upd, w_upd_),(v_upd, v_upd_),(h_upd, h_upd_)]
#update = [(w, w + w_upd_),(h, h + h_upd_),(v, v + v_upd_),(w_upd, w_upd_),(v_upd, v_upd_),(h_upd, h_upd_),(neg_sample, v_given_h_sample)]

cost = T.sum(T.sqr(x - v_given_h_p))*(1.0/MB_size)
#train = theano.function(inputs=[x], updates=update)

train = theano.function(
    inputs=[index],
    outputs=cost,
    updates=update,
    givens={
        x: data_x_shared[index * MB_size: (index + 1) * MB_size]
    }
)

num_MBs = data_x.shape[0]/MB_size
#print num_MBs
errs = np.zeros(num_MBs)

print w.get_value().sum()
for epoch in range(10):

    if epoch>5:
        momentum=finalmomentum;
    else:
        momentum=initialmomentum;

    for MB in range(num_MBs):
        #print epoch%num_MBs
        err = train(MB)
	errs[MB] = err
	#print MB,err
	#print w.get_value().sum()

	#print neg_samples.get_value(borrow=True)
    #print np.sum(w.get_value())
    if epoch%1 == 0:
        image = Image.fromarray(
            tile_raster_images(
                X=w.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(25, 40),
                tile_spacing=(1, 1)
            )
        )
        image.save('../filters1/%i.png' % epoch)
        #err = np.sum((data_x - neg_sample.get_value(borrow=True))**2)
        print epoch,errs.mean()*N

#train(x)

#img = np.zeros((28,28),dtype='uint8')
#image_data = train_set[0][0]
#image_data = image_data.reshape((28,28))
#for i in range(28):
#    for j in range(28):
#        img[i,j] = 255*image_data[i][j]
#image = Image.fromarray(img)
#image.save('samples.jpg')

