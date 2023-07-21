#!/usr/bin/python

import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio



def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other



def do_load(training_stages,layers,prob,maxit=300,r=1000,ivl=10):
    """
    training LITA network for Optimal Feedback Matrix Problem (OFMP)
    """

    ######## TIME PROFILE
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    ####  FOR TRAINING PARA. SAVE PERPOSE ####
    savefile=prob.name[0:-3]+'npz'
    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    return sess
