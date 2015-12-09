import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from utils import tile_raster_images

test_code = []

dataset_name = 'MNIST/2'
dataset = '../data/mnist.pkl.gz'
datafile = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(datafile)
datafile.close()

ax = plt.gca()
artists = []
zoom = 0.1
datax = test_set[0]
fig, ax = plt.subplots()
for i in range(len(test_set[0])):
	channel = 1 - datax[i][np.newaxis]
	img = Image.fromarray(tile_raster_images(X=(channel,channel,channel,None),img_shape=(28,28),tile_shape=(1,1)))
	img.save('test.png')
	img = plt.imread(get_sample_data('/home/llajan/RBM/from_scratch/test.png'))
	im = OffsetImage(img, zoom=zoom)
	ab = AnnotationBbox(im, (test_code[i][0], test_code[i][1]), xycoords='data', frameon=False)
	artists.append(ax.add_artist(ab))
ax.update_datalim(np.column_stack([test_code[:,0],test_code[:,1]]))
ax.autoscale()

plt.savefig('clusters.eps', format='eps', dpi=1000)
