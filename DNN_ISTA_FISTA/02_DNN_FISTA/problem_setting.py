#!/usr/bin/python
import tensorflow as tf
import scipy.io as sio



class TFGenerator(object):
    def __init__(self,F,**kwargs):
        self.F = F
        M,N = F.shape # for size of problem (input, output)
        vars(self).update(kwargs)
        self.x_ = tf.placeholder(tf.float32,(None, N*N+N*N+N*M),name='AB') # vectorized x=[vec(A)^T, vec(B1)^T, vec(B2)^T]^T: column vector form
        self.F_ = tf.placeholder(tf.float32,(None, M*N),name='Finit')
        self.y_ = tf.placeholder(tf.float32,(None, M*N),name='Fstar')



def control_dataset():
    """
    [M,N] = size(F)  , u(t) = -F x(t)
    r: Number of dataset
    """
    # filename='Mass_Spring_x_f_y_dataset_ver03_1000.mat'
    # filename='Ksparse_x_f_y_dataset_normalized_N5_1000.mat' # N=5 multi-agent system
    filename = 'Ksparse_x_f_y_dataset_N5_1000.mat' # N=10 multi-agent system

    # mat = sio.loadmat(filename)
    mat = sio.loadmat(filename)

    ###### Training dataset
    x_train_data = mat['x_train_data']  # vectorized x=[vec(A)^T, vec(B1)^T, vec(B2)^T]^T: column vector form
    y_train_data = mat['y_train_data']  # vec(Fstar): column vector form
    Finit_train_data = mat['Finit_train_data']  # LQR solution for the initial point of the algorithm
    F_LQR = mat['F_init']

    ######## Test dataset
    x_test_data = mat['x_test_data']
    y_test_data = mat['y_test_data']
    Finit_test_data = mat['Finit_test_data']

    # one random example for size setting
    F = F_LQR

    # generate an object
    prob = TFGenerator(F=F)
    prob.name = filename

    prob.xval = x_train_data
    prob.yval = y_train_data
    prob.Finit = Finit_train_data
    prob.xval_test = x_test_data
    prob.yval_test = y_test_data
    prob.Finit_test = Finit_test_data



    ###### Specific data for use
#    filename = 'Ksparse_x_f_y_dataset_normalized_ver06_1.mat'
#    mat = sio.loadmat(filename)
#    x_real_data = mat['x_real_data_normalized']  # vectorized x=[vec(A)^T, vec(B1)^T, vec(B2)^T]^T: column vector form
#    Finit_real_data = mat['Finit_real_data']  # LQR solution for the initial point of the algorithm
#    prob.xval_real = x_real_data
#    prob.Finit_real = Finit_real_data

    return prob

