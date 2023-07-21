

import tensorflow as tf
import scipy.io as sio
import numpy as np


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



def do_use(layers,prob):
    """
    Test LITA network for Optimal Feedback Matrix Problem (OFMP)
    """

    ######## TIME PROFILE
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    ####  FOR TRAINING PARA. SAVE PERPOSE ####
    savefile=prob.name[0:-3]+'npz'
    load_trainable_vars(sess,savefile) # must load AFTER the initializer


    name, Fhat_ = layers[-1][0:2]
    name, Fhat_1_ = layers[-2][0:2]
    name, Fhat_2_ = layers[-3][0:2]
    name, Fhat_3_ = layers[-4][0:2]
    name, Fhat_4_ = layers[-5][0:2]
    name, Fhat_5_ = layers[-6][0:2]
    name, Fhat_6_ = layers[-7][0:2]
    name, Fhat_7_ = layers[-8][0:2]
    name, Fhat_8_ = layers[-9][0:2]
    name, Fhat_9_ = layers[-10][0:2]
    name, Fhat_10_ = layers[-11][0:2]

    Fhat,Fhat_1,Fhat_2,Fhat_3,Fhat_4,Fhat_5,Fhat_6,Fhat_7,Fhat_8,Fhat_9,Fhat_10 = sess.run([Fhat_,Fhat_1_,Fhat_2_,Fhat_3_,Fhat_4_,Fhat_5_,Fhat_6_,Fhat_7_,Fhat_8_,Fhat_9_,Fhat_10_],  feed_dict={prob.x_: prob.xval_real, prob.F_: prob.Finit_real})
    sio.savemat("Fhat.mat",{"Fhat": Fhat,"Fhat_1":Fhat_1,"Fhat_2":Fhat_2,"Fhat_3":Fhat_3,"Fhat_4":Fhat_4,"Fhat_5":Fhat_5,"Fhat_6":Fhat_6,"Fhat_7":Fhat_7,"Fhat_8":Fhat_8,"Fhat_9":Fhat_9,"Fhat_10":Fhat_10})
