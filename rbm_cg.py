import theano
import numpy as np
import theano.tensor as T
import os
import gzip
import cPickle
from utils import tile_raster_images
import Image
from theano.tensor.shared_randomstreams import RandomStreams


class stackedRBMs(object):
	def __init__(
		self,
		rbms,
		input = None
	):
		#self.rbms = rbms
		self.set_params()
		self.input = input
		self.code_layers = len(rbms)
		self.num_layers  = 2*len(rbms)
		self.w = []
		self.b = []
		self.linear = []
		for rbm in rbms:
			self.w.append(theano.shared(value=rbm.w.get_value(),borrow=True))
			self.b.append(theano.shared(value=rbm.h.get_value(),borrow=True))
			self.linear.append(rbm.linear)
		for rbm in reversed(rbms):
			self.w.append(theano.shared(value=np.transpose(rbm.w.get_value()),borrow=True))
			self.b.append(theano.shared(value=rbm.v.get_value(),borrow=True))
			self.linear.append(rbm.linear)

	def fprop(self):
		data = self.input
		for layer in range(self.num_layers):
			#data = rbm.fprop(data)
			pre_sigmoid = T.dot(data,self.w[layer])+self.b[layer]
			if self.linear[layer]:
				data = pre_sigmoid
			else:
				data = T.nnet.sigmoid(pre_sigmoid)

		#code = data.copy()
		#for layer in range(self.num_layers-1,-1,-1):
		#	#data = rbm.fprop(data)
		#	data = T.nnet.sigmoid(T.dot(data,T.transpose(self.w[layer]))+self.h[layer])

		#return code, data
		return data

	def set_params(	
		self,
		lr 				= 0.1,
		weightcost  	= 0.0002,
		#MB_size			= 1000,
		momentum		= 0.5
	):
		self.lr = lr
		self.weightcost = weightcost
		#self.MB_size = MB_size
		self.momentum = momentum

	def get_cost_ent(self):
		pred = self.fprop()
		ent = T.nnet.binary_crossentropy(pred,self.input)
		#return T.sum(ent)*(1.0/self.MB_size)
		return T.mean(T.sum(ent,axis=1))
		
	def get_cost_sqr(self):
		pred = self.fprop()
		#cost = T.sum(T.sqr(self.input - pred))*(1.0/self.MB_size)
		cost = T.mean(T.sum(T.sqr(self.input - pred),axis=1))
		return cost

	def get_updates(self):
		cost = self.get_cost_ent()
		updates = []
		for i in range(self.num_layers):
			updates.append((self.w[i],self.w[i] - 0.01*T.grad(cost,self.w[i])))
			updates.append((self.b[i],self.b[i] - 0.01*T.grad(cost,self.b[i])))
		return updates

	#def fprop(self):
	#	data = self.input
	#	for layer in range(self.num_layers,-1,-1):
	#		#data = rbm.fprop(data)
	#		pre_sigmoid = T.dot(data,T.transpose(self.w[layer]))+self.v[layer]
	#		if self.linear:
	#			data = pre_sigmoid
	#		else:
	#			data = T.nnet.sigmoid(pre_sigmoid)

	#	return data


class RBM(object):
	def __init__(
		self,
		n_visible=784,
		n_hidden=1000,
		linear=False,
		input = None,
		w = None,
		h = None,
		v = None,
		numpy_rng=None,
		theano_rng=None
	):

		self.linear = linear

		self.n_visible = n_visible
		self.n_hidden  = n_hidden

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1234)
		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if w is None:
			initial_W = 0.1*np.asarray(np.random.randn(n_visible, n_hidden),dtype=theano.config.floatX)
			w = theano.shared(value=initial_W,borrow=True)
		if h is None:
			h = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX))
		if v is None:
			v = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX))

		self.input = input
		if not input:
			self.input = T.matrix('input')

		self.w,self.h,self.v = w,h,v
		self.theano_rng = theano_rng
		self.params = [self.w, self.h, self.v]

		self.w_upd = theano.shared(value=np.zeros((n_visible,n_hidden),dtype=theano.config.floatX))
		self.v_upd = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX))
		self.h_upd = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX))

		self.set_params()
	
	def set_params(	
		self,
		lr 				= 0.1,
		weightcost  	= 0.0002,
		MB_size			= 100,
		momentum		= 0.5
	):
		self.lr = lr
		self.weightcost = weightcost
		self.MB_size = MB_size
		self.momentum = momentum

	def fprop(self,inp):
		pre_sigmoid = T.dot(inp,self.w)+self.h
		if self.linear:
			return pre_sigmoid
		else:
			return T.nnet.sigmoid(pre_sigmoid)

	def bprop(self,inp):
		return T.nnet.sigmoid(T.dot(inp,T.transpose(self.w))+self.v)

	def get_upds(self,inp):
		w_update = T.dot(T.transpose(inp),self.fprop(inp))*(1.0/self.MB_size)
		h_update = T.mean(self.fprop(inp),axis=0)
		v_update = T.mean(inp,axis=0)
		return w_update,h_update,v_update
	
	def get_updates(self):

		h_given_v_p 		= self.fprop(self.input)
		if self.linear:
			h_given_v_sample = h_given_v_p + self.theano_rng.normal(size=h_given_v_p.shape,dtype=theano.config.floatX)
		else:
			h_given_v_sample = self.theano_rng.binomial(size=h_given_v_p.shape,n=1,p=h_given_v_p,dtype=theano.config.floatX)
			#h_given_v_sample = T.gt(h_given_v_p,self.theano_rng.uniform(size=h_given_v_p.shape))
		v_given_h_p 		= self.bprop(h_given_v_sample)
		#v_given_h_sample 	= self.theano_rng.binomial(n=1,p=v_given_h_p,dtype=theano.config.floatX)
	
		pos_w, pos_h, pos_v = self.get_upds(self.input)
		neg_w, neg_h, neg_v = self.get_upds(v_given_h_p)
		
		w_upd_ = self.momentum*self.w_upd + self.lr*((pos_w - neg_w) - self.weightcost*self.w)
		v_upd_ = self.momentum*self.v_upd + self.lr*(pos_v - neg_v)
		h_upd_ = self.momentum*self.h_upd + self.lr*(pos_h - neg_h)
		
		update = [
			(self.w, self.w + w_upd_),
			(self.h, self.h + h_upd_),
			(self.v, self.v + v_upd_),
			(self.w_upd, w_upd_),
			(self.v_upd, v_upd_),
			(self.h_upd, h_upd_)
		]
		#update = [(w, w + w_upd_),(h, h + h_upd_),(v, v + v_upd_),(w_upd, w_upd_),(v_upd, v_upd_),(h_upd, h_upd_),(neg_sample, v_given_h_sample)]
		
		cost = T.sum(T.sqr(self.input - v_given_h_p))*(1.0/self.MB_size)
		#train = theano.function(inputs=[x], updates=update)
	
		return cost, update


def test_rbm():

	dataset = '../data/mnist.pkl.gz'
	datafile = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(datafile)
	datafile.close()

	N = train_set[0].shape[0]
	d = train_set[0].shape[1]
	Nv = valid_set[0].shape[0]

	data_x = train_set[0]
	np.random.shuffle(data_x)
	data_x_shared = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
	valid_x_shared = theano.shared(np.asarray(valid_set[0], dtype=theano.config.floatX))

	lr 				= 0.1
	weightcost  	= 0.0002
	MB_size			= 100
	momentum		= 0.5
	initialmomentum	= 0.5
	finalmomentum  	= 0.9
	max_epochs	 	= 10
	momentumthresh 	= 5
	vis, hid1, hid2, hid3, hid4 = 784,1000,500,250,30
	num_MBs = data_x.shape[0]/MB_size


	x = T.matrix()
	index = T.lscalar()  # index to a [mini]batch

	rbm1 = RBM(input=x,n_visible=vis ,n_hidden=hid1)
	rbm2 = RBM(input=x,n_visible=hid1,n_hidden=hid2)
	rbm3 = RBM(input=x,n_visible=hid2,n_hidden=hid3)
	rbm4 = RBM(input=x,linear=True,n_visible=hid3,n_hidden=hid4)
	rbm4.set_params(lr=0.001,weightcost=0.0002)

	rbms = [rbm1, rbm2, rbm3, rbm4]
	data = [data_x_shared]
	img_size = [(25,40),(20,25),(10,25),(5,6)]

	#for rbm_i in range(4):

	#	rbm = rbms[rbm_i]	

	#	cost, update = rbm.get_updates()

	#	train = theano.function(
	#	    inputs=[index],
	#	    outputs=cost,
	#	    updates=update,
	#	    givens={
	#	        x: data[-1][index * MB_size: (index + 1) * MB_size]
	#	    }
	#	)

	#	errs = np.zeros(num_MBs)
	#	
	#	#print w.get_value().sum()
	#	for epoch in range(max_epochs):
	#	
	#		if epoch>momentumthresh:
	#			rbm.set_params(momentum=finalmomentum)
	#		#else:
	#		#	momentum=initialmomentum
	#	
	#		for MB in range(num_MBs):
	#	        #print epoch%num_MBs
	#			err = train(MB)
	#			errs[MB] = err
	#		#print MB,err
	#		#print w.get_value().sum()
	#	
	#		#print neg_samples.get_value(borrow=True)
	#	    #print np.sum(w.get_value())
	#		if epoch%1 == 0:
	#			#print img_size[rbm_i]
	#			#print rbm.w.get_value().shape
	#			#image = Image.fromarray(
	#	        #    tile_raster_images(
	#	        #        X=rbm.w.get_value(borrow=True).T,
	#	        #        img_shape=(28, 28),
	#			#		tile_shape=img_size[rbm_i],
	#	        #        tile_spacing=(1, 1)
	#	        #    )
	#	        #)
	#			#image.save('../filters/%i.png' % epoch)
	#			#err = np.sum((data_x - neg_sample.get_value(borrow=True))**2)

	#			print epoch,errs.mean()*N
	#			#print rbm.w.get_value().sum(), rbm.h.get_value().sum(), rbm.v.get_value().sum()

	#	data.append(rbm.fprop(data[-1]))

	#output = open('rbms.pkl','w')
	#cPickle.dump(rbms,output)
	#output.close()
	
	datafile = open('rbms.pkl', 'r')
	rbms = cPickle.load(datafile)
	datafile.close()

	sRBM = stackedRBMs(rbms=rbms,input=x)

	#get_code = theano.function(inputs=[],outputs=sRBM.fprop(),givens={x: data_x_shared})
	#code,recon = get_code()
	#img = Image.fromarray(tile_raster_images(X=recon,img_shape=(28,28),tile_shape=(100,100)))
	#img.save('img.png')

	MB_size = 100
	num_MBs = N/MB_size
	max_epochs = 500
	train_errors = []
	valid_errors = []
	errs = np.zeros(num_MBs)

	train = theano.function(
	    inputs=[index],
	    outputs=sRBM.get_cost_sqr(),
	    updates=sRBM.get_updates(),
	    givens={
	        x: data_x_shared[index * MB_size: (index + 1) * MB_size]
	    }
	)

	valid = theano.function(
	    inputs=[index],
	    outputs=sRBM.get_cost_sqr(),
	    givens={
	        x: valid_x_shared[index * Nv: (index + 1) * Nv]
	    }
	)

	
	#print w.get_value().sum()
	for epoch in range(max_epochs):
	
		if epoch>momentumthresh:
			sRBM.set_params(momentum=finalmomentum)
		#else:
		#	momentum=initialmomentum
	
		for MB in range(num_MBs):
	        #print epoch%num_MBs
			err = train(MB)
			errs[MB] = err
		#print MB,err
		#print w.get_value().sum()
	
		#print neg_samples.get_value(borrow=True)
	    #print np.sum(w.get_value())
		if epoch%1 == 0:
			#print img_size[rbm_i]
			#print rbm.w.get_value().shape
			#image = Image.fromarray(
	        #    tile_raster_images(
	        #        X=rbm.w.get_value(borrow=True).T,
	        #        img_shape=(28, 28),
			#		tile_shape=img_size[rbm_i],
	        #        tile_spacing=(1, 1)
	        #    )
	        #)
			#image.save('../filters/%i.png' % epoch)
			#err = np.sum((data_x - neg_sample.get_value(borrow=True))**2)

			valid_error = valid(0)
			print epoch,errs.mean(),valid_error
			train_errors.append(errs.mean())
			valid_errors.append(valid_error)
			#print rbm.w.get_value().sum(), rbm.h.get_value().sum(), rbm.v.get_value().sum()


	output = open('srbm.pkl','w')
	cPickle.dump(sRBM,output)
	cPickle.dump(train_errors,output)
	cPickle.dump(valid_errors,output)
	output.close()
	
	#train(x)
	
	#img = np.zeros((28,28),dtype='uint8')
	#image_data = train_set[0][0]
	#image_data = image_data.reshape((28,28))
	#for i in range(28):
	#    for j in range(28):
	#        img[i,j] = 255*image_data[i][j]
	#image = Image.fromarray(img)
	#image.save('samples.jpg')


	
test_rbm()


