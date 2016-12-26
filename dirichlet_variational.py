import numpy as np
from scipy.special import digamma
from scipy.special import beta
from sklearn.preprocessing import binarize
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# honestly, took that from stackoverflow for sake of visualization
def visualize_confusion_matrix(conf_arr):
	norm_conf = []
	for i in conf_arr:
		a = 0
		tmp_arr = []
		a = sum(i, 0)
		for j in i:
			tmp_arr.append(float(j) / float(a))
		norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap = plt.cm.jet, interpolation = 'nearest')

	width, height = conf_arr.shape

	for x in xrange(width):
		for y in xrange(height):
			ax.annotate(str(conf_arr[x][y]), xy = (y, x), horizontalalignment = 'center', verticalalignment = 'center')
	plt.rc('text', usetex = True)
	plt.rc('font', family = 'serif', size = 14)		
	
	cb = fig.colorbar(res)
	alphabet = '0123456789'
	plt.title('Confusion matrix')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.show()


class DPMixtire:
	def restart(self):
		self.rnk = np.random.uniform(low = 0.0, high = 1.0, size = (self.N, self.K))
		self.rnk = np.einsum('nk,n->nk', self.rnk, 1.0 / self.rnk.sum(axis = 1))

	def __init__(self, X, alpha, a, b):
		self.rnk = None
		self.rnk_best = None
		self.X = np.reshape(binarize(X, threshold = 8.0), (X.shape[0], 64))
		self.alpha = alpha
		self.a = a
		self.b = b
		
		self.N = X.shape[0]
		self.D = X.shape[1]
		self.K = int(10 * self.alpha * np.log(1.0 + self.N / self.alpha))
		
		self.restart()
		
		self.akv = None
		self.bkv = None
		
		self.alphakdtheta = None
		self.betakdtheta = None

	def log_average(self, a, b):
		return digamma(a) - digamma(a + b)

	def recompute_abkv(self):
		akv = self.rnk.sum(axis = 0)
		bkv = np.zeros(self.K)
		
		for t in range(self.K - 1):
			bkv[t] = akv[t + 1:].sum()
		self.akv = akv + 1.0
		self.bkv = bkv + self.alpha

	def recompute_alphabetakdtheta(self):
		self.alphakdtheta = self.a + np.einsum('nd,nk->dk', self.X, self.rnk)
		self.betakdtheta = self.b + np.einsum('nd,nk->dk', (1.0 - self.X), self.rnk)

	def compute_rhonk(self):
		# add \Braket{\log p(x_n|\theta_k)}_{q(\theta)}
		rhonk = np.einsum('nd,dk->nk', self.X, self.log_average(self.alphakdtheta, self.betakdtheta)) + \
				np.einsum('nd,dk->nk', (1.0 - self.X), self.log_average(self.betakdtheta, self.alphakdtheta))
		
		# add \Braket{\log v_k}_{q(v)}
		rhonk += np.einsum('n,k->nk', np.full(self.N, 1.0), self.log_average(self.akv, self.bkv))
		
		# add \sum\limits_{i = 1}^{k - 1} \Braket{\log (1 - v_i)_{q(v)}}
		logsum = np.zeros(self.N)
		for k in range(self.K):
			rhonk[:, k] += logsum
			logsum += self.log_average(self.bkv[k], self.akv[k])
		return rhonk	

	def recompute_rnk(self):
		exprhonk = np.exp(self.compute_rhonk())
		self.rnk = np.einsum('nk,n->nk', exprhonk, 1.0 / exprhonk.sum(axis = 1)) # normalize
	
	def get_pik(self):
		pik = np.zeros(self.K)
		current_prod = 1.0
		for k in range(self.K):
			vk = self.akv[k] / (self.akv[k] + self.bkv[k])
			pik[k] = current_prod * vk
			current_prod *= (1.0 - vk)
		return pik	

	def compute_loss(self):
		# add += \sum_{n k} r_{nk} \rho_{nk}
		loss = np.einsum('nk,nk', self.rnk, self.compute_rhonk())

		# add \sum_{k d} \Braket{\log p(\theta_k^d)}_{q(\theta)}
		loss += (-np.log(beta(self.a, self.b)) + \
					(self.a - 1.0) * self.log_average(self.alphakdtheta, self.betakdtheta) + \
					(self.b - 1.0) * self.log_average(self.betakdtheta, self.alphakdtheta)).sum()

		# add \sum_k \Braket{B(v_k|1, alpha)}_{q(v)}
		loss += (-np.log(beta(1.0, self.alpha)) + (self.alpha - 1.0) * self.log_average(self.akv, self.bkv)).sum()

		# subtract \Braket{\log q(\theta)}_{q(\theta)}
		loss -= (-np.log(beta(self.alphakdtheta, self.betakdtheta)) + \
					(self.alphakdtheta - 1.0) * self.log_average(self.alphakdtheta, self.betakdtheta) + \
					(self.betakdtheta - 1.0) * self.log_average(self.betakdtheta, self.alphakdtheta)).sum()

		# subtract \Braket{\log q(v)}_{q(v)}
		loss -= (-np.log(beta(self.akv, self.bkv)) + 
					(self.akv - 1.0) * self.log_average(self.akv, self.bkv) + \
					(self.bkv - 1.0) * self.log_average(self.bkv, self.akv)).sum()

		# subtract \Braket(\log q(Z))_{q(Z)}
		loss -= (self.rnk * np.log(self.rnk)).sum()

		return loss

	def add_sample(self, X_new):
		self.X  = np.concatenate((self.X, np.reshape(binarize(X_new, threshold = 8.0), (X_new.shape[0], 64))), axis = 0)
		self.N = self.X.shape[0]
		self.restart()
		return self.var_inference(max_iter = 11)

	def show_clusters(self):
		top_10_indices = self.get_pik().argsort()[-12:][::-1]
		top_thetas = self.alphakdtheta[:, top_10_indices] / (self.alphakdtheta[:, top_10_indices] + self.betakdtheta[:, top_10_indices])
		top_thetas = np.reshape(top_thetas.T, (12, 8, 8))
		plt.rc('text', usetex = True)
		plt.rc('font', family = 'serif', size = 14)	
		plt.title('Clusters centers')
		for i in range(12):
			plt.subplot(3, 4, i + 1)
			plt.imshow(top_thetas[i, :, :], cmap = 'gray')
		plt.show()

	def var_inference(self, num_start = 1, display = True, max_iter = 100, tol_L = 1e-4):
		converged = False
		loss_best = -10000000
		rnk_best = None

		for stard_id in range(num_start):
			loss = -10000000
			self.restart()
			iter_id = 0
			while not converged and iter_id < max_iter:
				iter_id += 1

				self.recompute_alphabetakdtheta()
				self.recompute_abkv()
				self.recompute_rnk()
				
				new_loss = self.compute_loss()
				if np.abs(new_loss - loss) < tol_L:
					converged = True
				if display:
					print 'loss = ' + str(self.compute_loss())
					print 'clusters left: ' + str(len(np.unique(self.rnk.argmax(axis = 1))))

				if display and iter_id % 10 == 0:
					self.show_clusters()
			if loss > loss_best:
				self.rnk_best = self.rnk
		return self		
X = load_digits()


### centers_search_full_data task ### (remove ''' ''' in order to test)
'''
solver = DPMixtire(X.data, 3, 3 * 0.4, 7 * 0.4)
solver.var_inference(max_iter = 31, display = True)
'''





### classificator task ### (remove ''' ''' in order to test)
'''
solver = DPMixtire(X.data, 3, 3 * 0.4, 7 * 0.4)
solver.var_inference(max_iter = 31, display = True)
import pandas as pd
from sklearn.cross_validation import train_test_split

# leave only nonempty clusters
rnk = solver.rnk_best
X_total = pd.DataFrame(rnk[:, np.unique(rnk.argmax(axis = 1))])
print X_total.shape
X_total['label'] = pd.Series(X.target, index = X_total.index)
X_train, X_test, y_train, y_test = train_test_split(X_total.drop('label', 1), 
														X_total['label'].values, test_size = 0.33, random_state = 42)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 6)
clf.fit(X_train, y_train)

pred_test = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
visualize_confusion_matrix(confusion_matrix(y_test, pred_test))
'''




### data_bunches task ### (remove ''' ''' in order to test)
'''
X = X.data
X_divided = np.array_split(X, 10, axis = 0)

solver = DPMixtire(X_divided[0], 3, 3 * 0.4, 7 * 0.4)
for i in range(1, 10):
	solver = solver.add_sample(X_divided[i])
'''
