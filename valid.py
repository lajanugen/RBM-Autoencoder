import theano
import numpy as np
#import theano.tensor as T
import os
import gzip
import cPickle
from utils import tile_raster_images
import Image
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt


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

		output = open('srbm.pkl','r')
		sRBM_ = cPickle.load(output)
		output.close()
		for i in range(len(self.w)):
			self.w[i] = theano.shared(value=sRBM_.w[i].get_value(),borrow=True)
			self.b[i] = theano.shared(value=sRBM_.b[i].get_value(),borrow=True)

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



output = open('srbm.pkl','r')
sRBM_ 	= cPickle.load(output)
tr1  	= cPickle.load(output)
va1  	= cPickle.load(output)
output.close()
output = open('srbm1.pkl','r')
sRBM_ 	= cPickle.load(output)
tr2  	= cPickle.load(output)
va2  	= cPickle.load(output)
output.close()

xa = range(len(tr1) + len(tr2))
tr = tr1 + tr2
va = va1 + va2
plt.plot(xa,tr,'r',xa,va,'b')
plt.savefig('err.png')
