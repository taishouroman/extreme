# -*- encoding: utf-8 -*-

"""Find extreme elements for 1d and 2d array.

Functions:
	_DerivB(X, axis) -> retval
	_DerivF(X, axis) -> retval
	Is1stDiffPeak(X) -> retval
	IsHessianDefiniteness(X) -> lmax, lmin
	IsExtreme(X, etype) -> Y
	IsExtreme(X, 'both') -> lmax, lmin
"""

import copy;
import numpy as np;
import scipy.signal;
import skimage.feature;
import warnings;

def _DerivB(X, axis):
	"""Calculate the 1st backward discrete difference along given axis.

	Usage:
	_DerivB(X, axis) -> retval

	Arguments:
		X    (numpy.ndarray): Input array
		axis (int):           The axis along which the difference is taken

	Returns:
		retval(numpy.ndarray): The 1st backward discrete difference.
		                       The shape is the same as X except along axis where the dimensionis smaller by 1.
		                       The type is the same as X.
	"""
	return -np.flip(np.diff(np.flip(X, axis), axis=axis), axis);


def _DerivF(X, axis):
	"""Calculate the 1st forward discrete difference along given axis.

	Usage:
		_DerivF(X, axis) -> retval

	Arguments:
		X    (numpy.ndarray): Input array
		axis (int):           The axis along which the difference is taken

	Returns:
		retval (numpy.ndarray): The 1st forward discrete difference.
		                        The shape is the same as X except along axis where the dimensionis smaller by 1.
		                        The type is the same as X.
	"""
	return np.diff(X, axis=axis);


def Is1stDiffPeak(X):
	"""Check for each element if it is peak.

	Usage:
		Is1stDiffPeak(X) -> retval

	Arguments:
		X (numpy.ndarray): Input array

	Returns:
		retval (numpy.ndarray): Binary array, True means that the element is a peak.
		                        The shape is the same as X, the type is bool.

	Details:
		In this function, the extreme is defined that the sign of the 1st forward discrete difference and backward differ, and the end of the array is always False.
		That is why it is rare that the 1st order partial derivative will be exactly 0 although it is prerequisite for the extreme that all the derivatives are 0.
	"""
	Y = np.ones(X.shape, dtype=bool);
	for i in range(X.ndim):
		padt = [(0, 0) for j in range(X.ndim)];
		padf = copy.deepcopy(padt);
		padf[i] = (0, 1);
		padb = copy.deepcopy(padt);
		padb[i] = (1, 0);

		fd = np.pad(_DerivF(X, i), tuple(padf), 'constant', constant_values=0);
		bd = np.pad(_DerivB(X, i), tuple(padb), 'constant', constant_values=0);
		extre = fd * bd < 0;
		Y = np.logical_and(Y, extre);
	return Y;


def IsHessianDefiniteness(X):
	"""Check for each element if the Hessian matrix is positive or negative definite.

	Usage:
		IsHessianDefiniteness(X) -> lmax, lmin

	Arguments:
		X (numpy.ndarray): Input array

	Returns:
		lmax (numpy.ndarray): Binary array, True means that the Hessian of the element is negative definite.
		                      The shape is the same as X, the type is bool.
		lmin (numpy.ndarray): Binary array, True means that the Hessian of the element is positive definite.
		                      The shape is the same as X, the type is bool.

	Details:
		An element might be maximal if Hessian matrix is negative definite, minimax if the matrix is positive definite.
		Hessian matrix is negative definite if all eigenvalues of the matrix is negative, positive definite if all of is positive.
		This function doesn't support 3 or more dimensional array.
	"""
	if X.ndim == 1:
		lmax = np.zeros(X.shape, dtype=bool);
		lmax[scipy.signal.argrelmax(X)[0]] = True;
		lmin = np.zeros(X.shape, dtype=bool);
		lmin[scipy.signal.argrelmin(X)[0]] = True;
	elif X.ndim == 2:
		Hrr, Hrc, Hcc = skimage.feature.hessian_matrix(X, 1.0, 'nearest', order='rc');
		e1, e2 = skimage.feature.hessian_matrix_eigvals(Hrr, Hrc, Hcc);
		lmax = np.logical_and(e1 < 0, e2 < 0);
		lmin = np.logical_and(0 < e1, 0 < e2);
	elif X.ndim == 3:
		warnings.warn('This function doesn\'t support to 3-dims array.\nThis function applies for each slice.', FutureWarning);
		lmax = np.ndarray(X.shape, dtype=bool);
		lmin = np.ndarray(X.shape, dtype=bool);
		for i, I in enumerate(X):
			lmax[i, :, :], lmin[i, :, :] = IsHessianDefiniteness(I);
	else:
		raise ValueError('X.ndim must be less than or equal to 3.');
	return lmax, lmin;


def IsExtreme(X, etype='both'):
	"""Check for each element is it is the extreme.

	Usage:
		IsExtreme(X, etype) -> Y
		IsExtreme(X, 'both') -> lmax, lmin

	Arguments:
		X     (numpy.ndarray): Input array
		etype (str):           Extreme type
		                       'both'  The maximal and minimal (different variables)
		                       'extre' The extreme
		                       'max'   The maximal
		                       'min'   The minimal

	Returns:
		Y    (numpy.ndarray): Binary array, True means that the element is the specified extreme.
		                      The shape is the same as X, the type is bool.
		lmax (numpy.ndarray): Binary array, True means that the element is the maximal.
		                      The shape is the same as X, the type is bool.
		lmin (numpy.ndarray): Binary array, True means that the element is the maximal.
		                      The shape is the same as X, the type is bool.
	"""
	etype = etype.lower();
	if etype not in {'both', 'extre', 'max', 'min'}:
		raise ValueError('etype must be an either of "both", "extre", "max", "min".');

	# Check the 1st derivative
	F = Is1stDiffPeak(X);

	# Check the Hessian for each element
	lmax, lmin = IsHessianDefiniteness(X);

	# Return the specified extreme
	if etype == 'both':
		lmax = np.logical_and(F, lmax);
		lmin = np.logical_and(F, lmin);
		return lmax, lmin;
	elif etype == 'extre':
		Y = np.logical_and(F, np.logical_or(lmax, lmin));
		return Y;
	elif etype == 'max':
		Y = np.logical_and(F, lmax);
		return Y;
	elif etype == 'min':
		Y = np.logical_and(F, lmin);
		return Y;
