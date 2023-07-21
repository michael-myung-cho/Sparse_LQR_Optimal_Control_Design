

import tensorflow as tf
import scipy.io as sio
import numpy as np
import train

def do_test(layers,sess,prob):
    """
    Test Learning ISTA (DNN-ISTA) network for sparse LQR optimal control design
    """
    M,N = prob.F.shape
    ############### TEST ##############
    """
    Calculating Normalized Mean Sqauared Error (NMSE)
    """
    Fstar_ = tf.reshape(prob.y_,[-1,N,M])
    Fstar_ = tf.transpose(Fstar_, perm=[0, 2, 1])
    Fstar_ = tf.reshape(Fstar_,[-1,N*M])

    name, Fhat_ = layers[-1][0:2]
    Fhat_mat_ = tf.reshape(Fhat_,[-1,N*M])

    #### Load best learning parameters ####
    savefile=prob.name[0:-3]+'npz'
    state = train.load_trainable_vars(sess,savefile)

    #### Calcualting NMSE ####
    Fhat = sess.run(Fhat_mat_, feed_dict={prob.y_: prob.yval_test, prob.x_: prob.xval_test, prob.F_: prob.Finit_test}) # result in matrix form
    Fstar = sess.run(Fstar_, feed_dict={prob.y_: prob.yval_test, prob.x_: prob.xval_test, prob.F_: prob.Finit_test}) # ground truth in matrix form

    NMSE = 0
    for i in range(len(Fhat)):
        rel_err = (np.square(np.linalg.norm(Fhat[i]-Fstar[i])))/(np.square(np.linalg.norm(Fhat[i])))
        NMSE = NMSE + rel_err
    NMSE = NMSE/len(Fhat)
    print('Normalized Mean Squared Error (NMSE) in test: ', NMSE)


    ############# FOR DEBUGGING ##############
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("./logs")
    # writer.add_graph(sess.graph)

    #
    # ############## SAVE RESULT into MAT file ##############
    # sio.savemat("result.mat", {"Fhat": Fhat,"Fhat_normal":Fhat_normal,"loss":loss,"Fstar":Fstar,"rhoStep": rhoStep,"rhoThre":rhoThre,"W1P": W1P,"W2P":W2P,"W3P":W3P,"W1L": W1L,"W2L":W2L,"W3L":W3L})


