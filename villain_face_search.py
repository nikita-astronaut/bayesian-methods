import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import transform
import skimage
import copy

idx = 0

# we believe that log(0) can only be multiplied by 0 in our code
# thus, to avoid nan, we replace log(0) with 0 in log(M)
def safe_log(M):
	M[M < 0.00000001] = 1.0
	return np.log(M)

def get_lpx_d_all(X, F, B, s):
	h = F.shape[0]
	w = F.shape[1]
	N = X.shape[2]
	H = X.shape[0]
	W = X.shape[1]
	result = np.zeros((H - h + 1, W - w + 1, N))

	for k in range(N):
		pre_res = -(1.0 / 2.0) * ((X[:, :, k] - B) ** 2).sum() / (s ** 2) - (1.0 / 2.0) * H * W * np.log(2 * np.pi * s * s)
		result[:, :, k] = pre_res
		for d_h in range(H - h + 1):
			for d_w in range(W - w + 1):
				X_overlap = X[d_h:(d_h + h), d_w:(d_w + w), k]
				result[d_h, d_w, k] += (1.0 / 2.0) * ((X_overlap - B[d_h:(d_h + h), d_w:(d_w + w)]) ** 2).sum() / (s ** 2)
				result[d_h, d_w, k] -= (1.0 / 2.0) * ((X_overlap - F) ** 2).sum() / (s ** 2)
	return result
			
##################################################################
#
# Calculates log(p(X_k|d_k, F, B, s)) for all images X_k in X and 
# all possible displacements d_k.
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of Gaussian Noise
#
# Output parameters:
#   
#   lpx_d_all ... (H-h+1) x (W-w+1) x N numpy.array, 
#                 lpx_d_all[dh,dw,k] - log-likelihood of 
#                 observing image X_k given that the villain's 
#                 face F is located at displacement (dh, dw)
#
##################################################################

	
def calc_L(X, F, B, s, A, q, useMAP = False):
	log_pX = get_lpx_d_all(X, F, B, s)
	log_A = safe_log(A)
	L = None

	if not useMAP:
		L = (log_pX * q).sum()
		for k in range(X.shape[2]):
			L += (log_A * q[:, :, k]).sum()
	else:
		L = 0.0
		for k in range(X.shape[2]):
			L += log_pX[int(q[0, k]), int(q[1, k]), k]
			L += log_A[int(q[0, k]), int(q[1, k])]
	if not useMAP:
		L -= (q * safe_log(q)).sum()
	return L / X.shape[2]

###################################################################
#
# Calculates the lower bound L(q, F, B, s, A) for the marginal log 
# likelihood
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on 
#         displacement of face in any image
#   q  ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array, 
#             q[dh,dw,k] - estimate of posterior of displacement 
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#0             2 x N numpy.array, 
#             q[0,k] - MAP estimates of dh for X_k 
#             q[1,k] - MAP estimates of dw for X_k 
#   useMAP ... logical, if true then q is a MAP estimates of 
#              displacement (dh,dw) of villain's face given image 
#              Xk 
#
# Output parameters:
#   
#   L ... 1 x 1, the lower bound L(q, F, B, s, A) for the marginal log 
#         likelihood
#
###################################################################

	
def e_step(X, F, B, s, A, useMAP = False):
	q = np.zeros((A.shape[0], A.shape[1], X.shape[2]))
	log_pX = get_lpx_d_all(X, F, B, s)
	log_pX = log_pX - log_pX.max()

	for k in range(X.shape[2]):
		denominator = (np.exp(log_pX[:, :, k]) * A).sum()
		q[:, :, k] = (np.exp(log_pX[:, :, k]) * A) / denominator
	if not useMAP:
		return q
	q_map = np.zeros((2, X.shape[2]))
	for k in range(X.shape[2]):
		argmax = q[:, :, k].argmax()
		q_map[0, k] = int(argmax / A.shape[1])
		q_map[1, k] = int(argmax % A.shape[1])
	return q_map	
##################################################################
#
# Given the current esitmate of the parameters, for each image Xk
# esitmates the probability p(d_k|X_k,F,B,s,A)
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on 
#         displacement of face in any image
#   useMAP ... logical, if true then q is a MAP estimates of 
#              displacement (dh,dw) of villain's face given image 
#              Xk 
#
# Output parameters:
#   
#   q  ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array, 
#             q[dh,dw,k] - estimate of posterior of displacement 
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#             2 x N numpy.array, 
#             q[0,k] - MAP estimates of dh for X_k 
#             q[1,k] - MAP estimates of dw for X_k 
###################################################################

	
def m_step(X, q, h, w, useMAP = False, bg_converged = False, B_old = None):
	A = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	H = X.shape[0]
	W = X.shape[1]
	N = X.shape[2]
	# / X.shape[2] cause the sum in denominator literally equals N --- number of images
	if not useMAP:
		A = q.sum(axis = 2) / N
	else:
		for k in range(N):
			A[int(q[0, k]), int(q[1, k])] += 1.0 / N

	F = np.zeros((h, w))		
	if not useMAP:
		# F --- average over all possible mask displacements
		for a in range(h):
			for b in range(w):
				F[a, b] = (q * X[a:(H - h + 1 + a), b:(W - w + 1 + b), :]).sum() / N
	else:
		for k in range(N):
			F += X[int(q[0, k]):(int(q[0, k]) + h), int(q[1, k]):(int(q[1, k]) + w), k] / N
	B = np.zeros((H, W))

	if not bg_converged:
		if not useMAP:
			# B --- average over all mask displacements that don't overlap (a, b)
			for a in range(H):
				for b in range(W):
					# mask contains zeros for positions of face corner that overlap (a, b)
					mask = np.zeros((H - h + 1, W - w + 1, N)) + 1.0
					mask[max(0, a - h + 1):(a + 1), max(0, b - w + 1):(b + 1), :] = 0.0
					denominator = (q * mask).sum()
					for k in range(N):
						mask[:, :, k] *= X[a, b, k]
					B[a, b] = (q * mask).sum() / denominator
		else:
			for k in range(N):
				mask = np.zeros((H, W)) + 1.0
				mask[int(q[0, k]):int(q[0, k] + h), int(q[1, k]):int(q[1, k] + w)] = 0.0
				B += (X[:, :, k] * mask) / N
	else:
		B = B_old

x	s = 0.0
	if not useMAP:
		for a in range(H - h + 1):
			for b in range(W - w + 1):
				C = copy.deepcopy(B)
				C[a:(a + h), b:(b + w)] = F
				
				for k in range(N):
					s += q[a, b, k] * ((X[:, :, k] - C) ** 2).sum() / (N * H * W)
	else:
		for k in range(N):
			C = copy.deepcopy(B)
			C[int(q[0, k]):int(q[0, k] + h), int(q[1, k]):int(q[1, k] + w)] = F
			s += ((X[:, :, k] - C) ** 2).sum() / (N * H * W)
	return F, B, np.sqrt(s), A
					
###################################################################
# 
# Estimates F, B, s, A given esitmate of posteriors defined by q
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   q ... if useMAP = False:
#             (H-h+1) x (W-w+1) x N numpy.array, 
#             q[dh,dw,k] - estimate of posterior of displacement 
#             (dh,dw) of villain's face given image Xk
#           if useMAP = True:
#             2 x N numpy.array, 
#             q[0,k] - MAP estimates of dh for X_k 
#             q[1,k] - MAP estimates of dw for X_k 
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#  useMAP ... logical, if true then q is a MAP estimates of 
#             displacement (dh,dw) of villain's face given image 
#             Xk 
#
# Output parameters:
#   
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on 
#         displacement of face in any image
###################################################################
	
def run_EM(X, h, w, F = None, B = None, s = None, A = None,
    tolerance = 0.001, max_iter = 50, useMAP = False):
	converged = False

	s = 100000.0
	F = np.random.random((h, w))
	B = X[:, :, 0]
	A = np.random.random((X.shape[0] - h + 1, X.shape[1] - w + 1))
	A /= A.sum()
	
	LL = []
	L = calc_L(X, F, B, s, A, e_step(X, F, B, s, A, useMAP), useMAP)
	LL.append(L)
	cur_iter = 0
	while not converged:
		q = e_step(X, F, B, s, A, useMAP)

		F, B, s, A = m_step(X, q, h, w, useMAP)

		plt.imshow(F, cmap = 'Greys')
		plt.savefig('./' + str(cur_iter) + '_F.pdf', cmap = 'gray')

		plt.imshow(B, cmap = 'Greys')
		plt.savefig('./' + str(cur_iter) + '_B.pdf', cmap = 'gray')
		cur_iter += 1
		newL = calc_L(X, F, B, s, A, q, useMAP)
		LL.append(newL)
		if np.abs(L - newL) < tolerance:
			print newL
			converged = True
			# plt.imshow(F, cmap = 'Greys')
			# plt.savefig('./' + str(cur_iter) + '_F_' + str(idx) + '.pdf', cmap = 'gray')

			# plt.imshow(B, cmap = 'Greys')
			# plt.savefig('./' + str(cur_iter) + '_B_' + str(idx) + '.pdf', cmap = 'gray')
		L = newL
	return F, B, s, A, LL		
###################################################################
# 
# Runs EM loop until the likelihood of observing X given current
# estimate of parameters is idempotent as defined by a fixed 
# tolerance
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#   F, B, s, A ... initial parameters (optional!)
#   F ... h x w numpy.array, estimate of villain's face
#   B ... H x W numpy.array, estimate of background
#   s ... 1 x 1, estimate of standart deviation of Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, estimate of prior on 
#         displacement of face in any image
#   tolerance ... parameter for stopping criterion
#   max_iter  ... maximum number of iterations
#   useMAP ... logical, if true then after E-step we take only 
#              MAP estimates of displacement (dh,dw) of villain's 
#              face given image Xk 
#    
#
# Output parameters:
#   
#   F, B, s, A ... trained parameters
#   LL ... 1 x (number_of_iters + 2) numpy.array, L(q,F,B,s,A) 
#          at initial guess, after each EM iteration and after 
#          final estimate of posteriors;
#          number_of_iters is actual number of iterations that was 
#          done
###################################################################


def run_EM_modified(X, h, w, F = None, B = None, s = None, A = None,
    tolerance = 0.001, bg_tolerance = 0.02, max_iter = 50, useMAP = False):
	converged = False
	bg_converged = False
	s = 100000.0
	F = np.random.random((h, w))
	B = X[:, :, 0]
	A = np.random.random((X.shape[0] - h + 1, X.shape[1] - w + 1))
	A /= A.sum()
	
	LL = []
	L = calc_L(X, F, B, s, A, e_step(X, F, B, s, A, useMAP = useMAP), useMAP = useMAP)
	LL.append(L)
	cur_iter = 0
	while not converged:
		q = e_step(X, F, B, s, A, useMAP = useMAP)

		F, B_n, s, A = m_step(X, q, h, w, useMAP = useMAP, bg_converged = bg_converged, B_old = B)

		bg_change_average = np.sqrt((B - B_n) ** 2) / (X.shape[0] * X.shape[1])

		if bg_change_average < bg_tolerance:
			bg_converged = True

		plt.imshow(F, cmap = 'Greys')
		plt.savefig('./' + str(cur_iter) + '_F.pdf', cmap = 'gray')

		plt.imshow(B, cmap = 'Greys')
		plt.savefig('./' + str(cur_iter) + '_B.pdf', cmap = 'gray')
		cur_iter += 1
		newL = calc_L(X, F, B, s, A, q, useMAP)
		LL.append(newL)
		if np.abs(L - newL) < tolerance:
			print newL
			converged = True
			# plt.imshow(F, cmap = 'Greys')
			# plt.savefig('./' + str(cur_iter) + '_F_' + str(idx) + '.pdf', cmap = 'gray')

			# plt.imshow(B, cmap = 'Greys')
			# plt.savefig('./' + str(cur_iter) + '_B_' + str(idx) + '.pdf', cmap = 'gray')
		L = newL
	return F, B, s, A, LL

def run_EM_with_restarts(X, h, w, tolerance = 0.001, max_iter = 50,
                     useMAP = False, restart = 10):
	best_s = None
	best_F = None
	best_B = None
	best_A = None
	best_L = 100000000

	for rest in range(restart):
		F, B, s, A, LL = run_EM(X, h, w, tolerance , max_iter, useMAP)
		if LL[-1] < best_L:
			best_L = LL[-1]
			best_A, best_B, best_s, best_F = A, B, s, F
	return best_F, best_B, best_s, best_A, best_L
###################################################################
# 
# Restarts EM several times from different random initializations 
# and stores the best estimate of the parameters as measured by 
# the L(q,F,B,s,A)
#
# Input parameters:
#
#   X ... H x W x N numpy.array, N images of size H x W
#   h ... 1 x 1, face mask hight
#   w ... 1 x 1, face mask widht
#   tolerance, max_iter, useMAP ... parameters for EM
#   restart   ... number of EM runs
#
# Output parameters:
#   
#   F ... h x w numpy.array, the best estimate of villain's face
#   B ... H x W numpy.array, the best estimate of background
#   s ... 1 x 1, the best estimate of standart deviation of 
#         Gaussian noise
#   A ... (H-h+1) x (W-w+1) numpy.array, the best estimate of 
#         prior on displacement of face in any image
#   LL ... 1 x 1, the best L(q,F,B,s,A)
###################################################################

def gen_test_X(N, background, face, H, W, h, w, noise):
	B = io.imread(background)
	F = io.imread(face)
	F = skimage.color.rgb2gray(F)
	B = skimage.color.rgb2gray(B)
	F = transform.resize(F, (h, w))
	B = transform.resize(B, (H, W))
	
	X = np.zeros((H, W, N))

	for k in range(N):
		d_x = np.random.randint(0, H - h + 1)
		d_y = np.random.randint(0, W - w + 1)
		X[:, :, k] = B
		X[d_x:(d_x + h), d_y:(d_y + w), k] = F
		X[:, :, k] += np.random.normal(0, noise, (H, W))
	return X	


X = np.load('./dataShad300_2016.npy')
# X = gen_test_X(100, './bg.jpg', './face.jpeg', 100, 100, 40, 40, 0.2)
# run_EM(X, 40, 40, useMAP = True)
run_EM_modified(X[:, :, 0:200], 100, 73)
