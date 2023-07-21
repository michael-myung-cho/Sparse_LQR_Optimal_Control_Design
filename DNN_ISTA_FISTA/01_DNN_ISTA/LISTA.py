#!/usr/bin/python
"""
Building Learning ISTA (DNN-ISTA) network for sparse LQR optimal control design

By Myung (Michael) Cho
"""
import numpy as np
import numpy.linalg as la
import tensorflow as tf
import shrinkage



def compute_Lyapunov_org(A_,Q_):
    def tf_kron(a_,b_):
        a_shape = [a_.get_shape().as_list()[1],a_.get_shape().as_list()[2]]
        b_shape = [b_.get_shape().as_list()[1],b_.get_shape().as_list()[2]]

        temp_ = tf.reshape(a_, [-1, a_shape[0], 1, a_shape[1], 1]) * tf.reshape(b_, [-1, 1, b_shape[0], 1, b_shape[1]])
        kron_ab_= tf.reshape(temp_,[-1, a_shape[0]*b_shape[0], a_shape[1]*b_shape[1]])
        return kron_ab_

    u,M,N = A_.get_shape().as_list()
    I_ = tf.reshape(tf.eye(N),[-1,N,N])

    IA_ = tf_kron(I_, A_)
    AI_ = tf_kron(A_, I_)
    LHSlya_ = IA_ + AI_

    Qvec_ = tf.reshape(-Q_,[-1,N*N,1])
    P_ = tf.reshape(tf.matmul(tf.linalg.inv(LHSlya_),Qvec_),[-1,N,N])

    return P_


def Fmin_stateCheck(A_,B1_,B2_,R_,Q_,initial_rho,initial_gamma,Fhatpre_,GradJ_,rhoStep_,rhoThre_):

    eta = shrinkage.simple_soft_threshold
    Fhat_ = eta(Fhatpre_ - tf.scalar_mul(rhoStep_, GradJ_), rhoThre_)
    def stable_cond(Fhat_):
        max_eign_ = tf.math.reduce_max(tf.math.real(tf.py_func(np.linalg.eigvals, [A_ - tf.matmul(B2_, Fhat_)], tf.complex64)), axis=1)
        return max_eign_ >= 0
    return tf.where(stable_cond(Fhat_), Fhatpre_, Fhat_)


def build_LISTA(prob,T,initial_gamma,initial_rho,untied=False):
    """
    Builds a Learning ITA network for Sparse feedback matrix K
    return a list of layer info (name,xhat,newvars)
        - name : description, e.g. 'LITA T=1'
        - Fhat_ : that which approximates F_ at some point in the algorithm
        - newvars : a tuple of layer-specific trainable variables

        - dataset: vectorized x=[vec(A)^T, vec(B1)^T, vec(B2)^T]^T
    """

    assert not untied,'TODO: untied'
    layers = []  # type: list
    F = prob.F

    M,N = F.shape
    # Avec_, B1vec_, B2vec_ = tf.split(prob.x_,[N*N,N*M,N*M],1) # Mass spring case
    Avec_, B1vec_, B2vec_ = tf.split(prob.x_,[N*N,N*N,N*M],1) # Multi-agent case
    A_ = tf.reshape(Avec_,[-1,N,N])
    A_ = tf.transpose(A_, perm=[0, 2, 1])

    # B1_ = tf.reshape(B1vec_,[-1,M,N]) # Mass Spring case
    B1_ = tf.reshape(B1vec_,[-1,N,N]) # Multi-agent case
    B1_ = tf.transpose(B1_, perm=[0, 2, 1])

    B2_ = tf.reshape(B2vec_,[-1,M,N])
    B2_ = tf.transpose(B2_, perm=[0, 2, 1])

    Q_ = tf.constant(np.eye(N),dtype=tf.float32)
    R_ = tf.constant(np.eye(N),dtype=tf.float32)

    Fhat_ = tf.reshape(prob.F_, [-1,N,M]) # initial F
    Fhat_ = tf.transpose(Fhat_, perm=[0, 2, 1])

    for t in range(0,T):
        Fhatpre_ = Fhat_
        with tf.name_scope("layer"+str(t)):
            ############# Compute Lyapunov equation for P ##############
            APtemp_ = tf.transpose(A_ - tf.matmul(B2_, Fhatpre_),perm=[0,2,1])
            QPtemp_ = Q_ + tf.matmul(tf.transpose(Fhatpre_, perm=[0, 2, 1]), Fhatpre_)
            P_ = compute_Lyapunov_org(APtemp_, QPtemp_)


            ############# Compute Lyapunov equation for L ##############
            ALtemp_ = A_ - tf.matmul(B2_, Fhatpre_)
            QLtemp_ = tf.matmul(B1_,tf.transpose(B1_,perm=[0,2,1]))
            L_ = compute_Lyapunov_org(ALtemp_, QLtemp_)


            ############# Compute gradient ##############
            RFhat_ = Fhatpre_ # tf.matmul(R_, Fhatpre_)   # Since R=I
            B2transP_ = tf.matmul(tf.transpose(B2_,perm=[0,2,1]), P_)
            GradJ_ = 2 * tf.matmul(RFhat_ - B2transP_, L_)


            ############# Sparsity operation & Stability check ##############
            rhoStep_ = tf.Variable(1/initial_rho, name='rhoStep_{0}'.format(t), dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
            rhoThre_ = tf.Variable(initial_gamma/initial_rho, name='rhoThre_{0}'.format(t), dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
            Fhat_ = Fmin_stateCheck(A_,B1_,B2_,R_,Q_, initial_rho, initial_gamma, Fhatpre_, GradJ_,rhoStep_,rhoThre_)

            layers.append(('LISTA T='+str(t), Fhat_))

    return layers



