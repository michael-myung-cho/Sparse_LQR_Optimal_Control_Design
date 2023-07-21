#!/usr/bin/python

import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio



def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

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



def setup_training(layers,prob, trinit=1e-3):
    """
    training setup
    """

    training_stages=[]
    M,N = prob.F.shape
    Fstar_tmp_ = tf.reshape(prob.y_,[-1,N,M])
    Fstar_ = tf.transpose(Fstar_tmp_, perm=[0, 2, 1])
    Fstar_ = tf.reshape(Fstar_,[-1,N*M])

    ### loss with every layer
    # for name, Fhat_ in layers:
    #     Fhat_norm_ = tf.reshape(tf.norm(Fhat_, ord='fro', axis=[-2, -1]), [-1, 1])
    #     Fhat_mat_ = tf.reshape(Fhat_, [-1, N * M])
    #     Fhat_normal_ = Fhat_mat_ / Fhat_norm_
    #     loss_ = tf.reduce_mean(tf.square(Fhat_normal_ - Fstar_))
    #     train_ = tf.train.AdamOptimizer(learning_rate=trinit).minimize(loss_)
    #     training_stages.append((name,Fhat_,loss_,train_))

    ### loss only with last layer
    name, Fhat_ = layers[-1][0:2]
    #Fhat_norm_ = tf.reshape(tf.norm(Fhat_, ord='fro', axis=[-2, -1]),[-1,1])
    Fhat_mat_ = tf.reshape(Fhat_,[-1,N*M])
    #Fhat_normal_ = Fhat_mat_/Fhat_norm_
    loss_ = tf.reduce_mean(tf.square(Fhat_mat_ - Fstar_))
    train_ = tf.train.AdamOptimizer(learning_rate=trinit).minimize(loss_)
    training_stages.append((name,Fhat_,loss_,train_))
    return training_stages


def do_training(training_stages,layers,prob,maxit,r,ivl=10):
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
    done=state.get('done',[])
    log=str(state.get('log',''))


    for name, Fhat_,loss_,train_ in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            break

        print(name + ' ' + 'layer')
        Loss_hist = []
        Loss_best = 10000
        for i in range(maxit+1):
            if i%ivl == 0:
                Loss = sess.run(loss_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval, prob.F_: prob.Finit})

                ####  FOR TRAINING PARA. SAVE PERPOSE ####
                if Loss_best > Loss:
                    done = np.append(done, name)
                    log = log + '\n{name} loss={loss:.6f} dB in {i} iterations'.format(name=name, loss=Loss_best, i=i)
                    state['done'] = done
                    state['log'] = log
                    save_trainable_vars(sess, savefile, **state)

                if np.isnan(Loss):
                    raise RuntimeError('Loss is NaN')
                Loss_hist = np.append(Loss_hist,Loss)
                Loss_best = Loss_hist.min()
                sys.stdout.write('\r i={i:<6d} loss={loss:.6f} (best={best:.6f})\n'.format(i=i,loss=Loss,best=Loss_best))
                sys.stdout.flush()
            sess.run(train_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval, prob.F_: prob.Finit})

    # name,Fhat_,loss_,train_ = training_stages[-1]
    # print(name + ' ' + 'layer')
    # Loss_hist=[]
    # batch_size = int(r*1)
    # for epoch in range(maxit):
    #     total_batch = int(r/batch_size)
    #     for i in range(total_batch):
    #         x_batch = prob.xval[i * batch_size:(i + 1) * batch_size - 1, :]
    #         y_batch = prob.yval[i * batch_size:(i + 1) * batch_size - 1, :]
    #         Finit_batch = prob.Finit[i * batch_size:(i + 1) * batch_size - 1, :]
    #
    #         if i%ivl == 0:
    #             Loss = sess.run(loss_, feed_dict={prob.y_: y_batch, prob.x_: x_batch, prob.F_: Finit_batch})
    #             if np.isnan(Loss):
    #                 raise RuntimeError('Loss is NaN')
    #             Loss_hist = np.append(Loss_hist,Loss)
    #             Loss_best = Loss_hist.min()
    #             sys.stdout.write('\r epoch={epoch:<6d} i={i:<6d} loss={loss:.6f} (best={best:.6f})\n'.format(epoch=epoch,i=i,loss=Loss,best=Loss_best))
    #             sys.stdout.flush()
    #         sess.run(train_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval, prob.F_: prob.Finit})


    return sess
