import theano
import numpy as np
import theano.tensor as T
import os
import gzip
import cPickle
from utils import tile_raster_images
from utils import channel_image
import Image
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

class stackedRBMs(object):
	def __init__(
		self,
		rbms = None,
		params_file = None,
		input = None
	):
		self.set_params()
		self.input = input
		self.w = []
		self.b = []
		self.linear = []
		self.gaussian = []

		self.numpy_rng = np.random.RandomState(1234)
		self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
		
		if params_file:
			output = open(params_file,'r')
			sRBM_ = cPickle.load(output)
			output.close()

			self.num_layers = len(sRBM_.w)
			self.code_layers  = self.num_layers/2
			self.linear = sRBM_.linear
			self.gaussian = sRBM_.linear
			for i in range(self.num_layers):
				#self.w[i] = theano.shared(value=sRBM_.w[i].get_value(),borrow=True)
				#self.b[i] = theano.shared(value=sRBM_.b[i].get_value(),borrow=True)
				self.w.append(theano.shared(value=sRBM_.w[i].get_value(),borrow=True))
				self.b.append(theano.shared(value=sRBM_.b[i].get_value(),borrow=True))
		else:
			self.code_layers = len(rbms)
			self.num_layers  = 2*len(rbms)
			for rbm in rbms:
				self.w.append(theano.shared(value=rbm.w.get_value(),borrow=True))
				self.b.append(theano.shared(value=rbm.h.get_value(),borrow=True))
				self.linear.append(rbm.linear)
				self.gaussian.append(rbm.gaussian)
			for rbm in reversed(rbms):
				self.w.append(theano.shared(value=np.transpose(rbm.w.get_value()),borrow=True))
				self.b.append(theano.shared(value=rbm.v.get_value(),borrow=True))
				self.linear.append(rbm.linear)
				self.gaussian.append(rbm.gaussian)

		if rbms and self.use_momentum:
			self.w_upd = []
			self.b_upd = []
			for rbm in rbms:
				self.w_upd.append(theano.shared(value=np.zeros(rbm.w.get_value().shape,dtype=theano.config.floatX)))
				self.b_upd.append(theano.shared(value=np.zeros(rbm.h.get_value().shape,dtype=theano.config.floatX)))
			for rbm in reversed(rbms):
				self.w_upd.append(theano.shared(value=np.zeros(np.transpose(rbm.w.get_value()).shape,dtype=theano.config.floatX)))
				self.b_upd.append(theano.shared(value=np.zeros(rbm.v.get_value().shape,dtype=theano.config.floatX)))

		
	def fprop(self):
		data = self.input
		for layer in range(self.num_layers):
			#data = rbm.fprop(data)
			pre_sigmoid = T.dot(data,self.w[layer])+self.b[layer]
			if self.linear[layer]:
				data = pre_sigmoid
			elif self.gaussian[layer] and layer > 1:
				data = pre_sigmoid + self.theano_rng.normal(size=self.b[layer].shape,dtype=theano.config.floatX)
			else:
				data = T.nnet.sigmoid(pre_sigmoid)
				#mask = self.theano_rng.binomial(size=pre_sigmoid.shape,n=1,p=0.5,dtype=theano.config.floatX)
				#data = data*mask
			
		return data

	def set_params(	
		self,
		lr 				= 0.0001,
		weightcost  	= 0.0002,
		momentum		= 0.2,
		use_momentum    = False
	):
		self.lr = lr
		self.weightcost = weightcost
		self.momentum = momentum
		self.use_momentum = use_momentum

	def get_cost_ent(self):
		pred = self.fprop()
		ent = T.nnet.binary_crossentropy(pred,self.input)
		return T.mean(T.sum(ent,axis=1))
		
	def get_code(self):
		data = self.input
		for layer in range(self.code_layers):
			pre_sigmoid = T.dot(data,self.w[layer])+self.b[layer]
			if self.linear[layer]:
				data = pre_sigmoid
			else:
				data = T.nnet.sigmoid(pre_sigmoid)
		return data

	def get_cost_sqr(self):
		pred = self.fprop()
		cost = T.mean(T.sum(T.sqr(self.input - pred),axis=1))
		return cost

	def get_updates(self):
		#cost = self.get_cost_ent()
		cost = self.get_cost_sqr()
		updates = []
		for i in range(self.num_layers):
			if self.use_momentum:
				self.w_upd_ = self.momentum*self.w_upd[i] + self.lr*T.grad(cost,self.w[i])
				self.b_upd_ = self.momentum*self.b_upd[i] + self.lr*T.grad(cost,self.b[i])
				updates.append((self.w[i],self.w[i] - self.w_upd_))
				updates.append((self.b[i],self.b[i] - self.b_upd_))
				updates.append((self.w_upd[i],self.w_upd_))
				updates.append((self.b_upd[i],self.b_upd_))
			else:
				updates.append((self.w[i],self.w[i] - self.lr*T.grad(cost,self.w[i])))
				updates.append((self.b[i],self.b[i] - self.lr*T.grad(cost,self.b[i])))
		return updates


class RBM(object):
	def __init__(
		self,
		n_visible=784,
		n_hidden=1000,
		linear=False,
		gaussian=False,
		input = None,
		w = None,
		h = None,
		v = None,
		numpy_rng=None,
		theano_rng=None
	):

		self.linear = linear
		self.gaussian = gaussian
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
		if self.gaussian:
			return T.dot(inp,T.transpose(self.w))+self.v + self.theano_rng.normal(size=self.v.shape,dtype=theano.config.floatX)
		else:
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
		
		cost = T.sum(T.sqr(self.input - v_given_h_p))*(1.0/self.MB_size)
	
		return cost, update


# Load data
dataset_name = 'yalefaces/2'
dataset = '../data/yalefaces_aug.pkl'
datafile = open(dataset, 'r')
yalefaces = cPickle.load(datafile)
yalefaces = yalefaces - np.mean(yalefaces,axis=0)
yalefaces = yalefaces/np.std(yalefaces,axis=0)
train_set = yalefaces[:20000]
valid_set = yalefaces[20000:21000]
test_set = yalefaces[21000:21500]
#train_set, valid_set, test_set = cPickle.load(datafile)
datafile.close()

#N = train_set[0].shape[0]
#Nv = valid_set[0].shape[0]
#Nt = test_set[0].shape[0]
N = train_set.shape[0]
Nv = valid_set.shape[0]
Nt = test_set.shape[0]

# Shuffle data
#data_x = train_set[0]
data_x = train_set
np.random.shuffle(data_x)

# Shared variables for train,valid,test sets
data_x_shared = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
#valid_x_shared = theano.shared(np.asarray(valid_set[0], dtype=theano.config.floatX))
#test_x_shared = theano.shared(np.asarray(test_set[0], dtype=theano.config.floatX))
valid_x_shared = theano.shared(np.asarray(valid_set, dtype=theano.config.floatX))
test_x_shared = theano.shared(np.asarray(test_set, dtype=theano.config.floatX))

# Theano input variables
x = T.matrix()
index = T.lscalar()  # index to a [mini]batch

def train_RBMs():

	lr 				= 0.0001
	weightcost  	= 0.0002
	MB_size			= 100
	initialmomentum	= 0.5
	finalmomentum  	= 0.9
	max_epochs	 	= [200,50,20,20]
	momentumthresh 	= 5
	#vis, hid1, hid2, hid3, hid4 = 784,2000,500,250,3
	vis, hid1, hid2, hid3, hid4 = 2016,2500,600,300,100

	num_MBs = data_x.shape[0]/MB_size

	rbm1 = RBM(input=x,gaussian=True,n_visible=vis ,n_hidden=hid1)
	rbm2 = RBM(input=x,n_visible=hid1,n_hidden=hid2)
	rbm3 = RBM(input=x,n_visible=hid2,n_hidden=hid3)
	rbm4 = RBM(input=x,linear=True,n_visible=hid3,n_hidden=hid4) #Top layer is linear
	rbm1.set_params(lr=0.001,weightcost=0.0002)
	rbm2.set_params(lr=0.001,weightcost=0.0002)
	rbm3.set_params(lr=0.001,weightcost=0.0002)
	rbm4.set_params(lr=0.001,weightcost=0.0002)

	rbms = [rbm1, rbm2, rbm3, rbm4]
	data = [data_x_shared]

	# Plot image sizes
	img_size = [(50,50),(20,25),(10,25),(5,6)]

	# Train the four layers sequentially
	count = -1
	for rbm_i in range(4):

		rbm = rbms[rbm_i]	

		cost, update = rbm.get_updates()

		train = theano.function(
		    inputs=[index],
		    outputs=cost,
		    updates=update,
		    givens={
		        x: data[-1][index * MB_size: (index + 1) * MB_size]
		    }
		)

		errs = np.zeros(num_MBs)
		
		count += 1
		for epoch in range(max_epochs[count]):
		
			if epoch>momentumthresh:
				rbm.set_params(momentum=finalmomentum)
			#else:
			#	momentum=initialmomentum
		
			for MB in range(num_MBs):
				err = train(MB)
				errs[MB] = err
		
			if epoch%1 == 0:
				#Plot reconstructions (Applicable only to first layer)
				#image = Image.fromarray(
		        #    tile_raster_images(
		        #        X=rbm.w.get_value(borrow=True).T,
		        #        img_shape=(48, 42),
				#		tile_shape=img_size[rbm_i],
		        #        tile_spacing=(1, 1)
		        #    )
		        #)
				#image.save('../filters/%i.png' % epoch)
				#err = np.sum((data_x - neg_sample.get_value(borrow=True))**2)

				print epoch,errs.mean()*N

		data.append(rbm.fprop(data[-1]))

	# Dump parameters
	output = open(dataset_name + '/rbms.pkl','w')
	cPickle.dump(rbms,output)
	output.close()
	
def train_stackedRBMs():

	# Load pre-trained parameters
	datafile = open(dataset_name + '/rbms.pkl', 'r')
	rbms = cPickle.load(datafile)
	datafile.close()

	sRBM = stackedRBMs(rbms=rbms,input=x)

	#Plot reconstructions after pre-training
	#get_code = theano.function(inputs=[],outputs=sRBM.fprop(),givens={x: data_x_shared})
	#code,recon = get_code()
	#img = Image.fromarray(tile_raster_images(X=recon,img_shape=(28,28),tile_shape=(100,100)))
	#img.save('img.png')

	MB_size = 100
	num_MBs = N/MB_size
	max_epochs = 1000
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

	momentumthresh 	= 5
	finalmomentum 	= 0.9
	for epoch in range(max_epochs):
	
		#if epoch>momentumthresh:
		#	sRBM.set_params(momentum=finalmomentum)
		#else:
		#	momentum=initialmomentum
	
		for MB in range(num_MBs):
			err = train(MB)
			errs[MB] = err
	
		if epoch%1 == 0:
			valid_error = valid(0)
			print epoch,errs.mean(),valid_error
			train_errors.append(errs.mean())
			valid_errors.append(valid_error)

	# Dump params
	output = open(dataset_name + '/srbm.pkl','w')
	cPickle.dump(sRBM,output)
	cPickle.dump(train_errors,output)
	cPickle.dump(valid_errors,output)
	output.close()
	
def imscatter(x, y, image, ax, artists, zoom=1):
	#if ax is None:
	#x, y = np.atleast_1d(x, y)
	#artists = []
	return artists
	
def test_model():

	# Load pre-trained parameters
	#datafile = open(dataset_name + '/srbm.pkl', 'r')
	#srbm = cPickle.load(datafile)
	#datafile.close()

	sRBM = stackedRBMs(params_file=dataset_name + '/srbm.pkl',input=x)

	train = theano.function(
	    inputs=[index],
	    outputs=[sRBM.get_cost_sqr(),sRBM.fprop(),sRBM.get_code()],
	    givens={
	        x: data_x_shared[index * N: (index + 1) * N]
	    }
	)

	test = theano.function(
	    inputs=[index],
	    outputs=[sRBM.get_cost_sqr(),sRBM.fprop(),sRBM.get_code()],
	    givens={
	        x: test_x_shared[index * Nt: (index + 1) * Nt]
	    }
	)

	test_error, test_recon, test_code = test(0)
 	print test_error
	img = Image.fromarray(tile_raster_images(X=test_recon,img_shape=(48,42),tile_shape=(20,25)))
	img.save(dataset_name + '/recon.png')

	#rs = cm.rainbow(np.linspace(0, 1, 10))
	#print test_code.shape
	#for i in range(10):
	#	ix = test_code[np.where(test_set[1] == i)]
	#	#print ix
	#	plt.scatter(ix[:,0],ix[:,1],color=rs[i])
	##test_set[1]
	#plt.savefig(dataset_name + '/clusters.png')

	train_error, train_recon, train_code = train(0)
	datax,test_code = train_set,train_code

	ax = plt.gca()
	artists = []
	zoom = 0.3
	#datax = test_set
	fig, ax = plt.subplots()
	for i in range(N):
		#inds = np.where(test_set[1] == i)
		#ix = test_code[inds]
		#print ix
		#imscatter(ix[:,0], ix[:,1], datax, zoom = 0.5, ax = ax, artists = artists)
		#x_ = ix[:,0]
		#y_ = ix[:,1]
		#i = 0
		#for x0, y0 in zip(x_, y_):

		#img = Image.fromarray(tile_raster_images(X=(1 - datax[i][np.newaxis],None,None,None),img_shape=(28,28),tile_shape=(1,1)))
		channel = datax[i][np.newaxis]
		img = Image.fromarray(tile_raster_images(X=(channel,channel,channel,channel),img_shape=(48,42),tile_shape=(1,1)))
		#img = Image.fromarray(channel_image(X=1 - datax[i][np.newaxis],img_shape=(28,28),tile_shape=(1,1)))
		img.save('test.png')
		img = plt.imread(get_sample_data('/home/llajan/RBM/from_scratch/test.png'))
		im = OffsetImage(img, zoom=zoom)
		ab = AnnotationBbox(im, (test_code[i][0], test_code[i][1]), xycoords='data', frameon=False)
		artists.append(ax.add_artist(ab))
	ax.update_datalim(np.column_stack([test_code[:,0],test_code[:,1]]))
	ax.autoscale()

		#ax.scatter(ix[:,0],ix[:,1])
		#plt.scatter(ix[:,0],ix[:,1],color=rs[i])
	#test_set[1]
	plt.savefig(dataset_name + '/clusters1.eps', format='eps', dpi=1000)
	#plt.savefig(dataset_name + '/clusters1.png')



#test_model()
#train_RBMs()
#train_stackedRBMs()
test_model()

