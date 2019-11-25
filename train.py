import tensorflow as tf
import numpy as np
import numpy.random as rng
import os

class Trainer:
    """
    Training class for the standard MADEs/MAFs classes using a tensorflow optimizer.
    """
    def __init__(self, model,  args=None):
        """
        Constructor that defines the training operation.
        :param model: made/maf instance to be trained.
        :param optimizer: tensorflow optimizer class to be used during training.
        :param optimizer_arguments: dictionary of arguments for optimizer intialization.
        """
        self.early_stopping_count = 0
        self.model_parms = None
        self.model = model
        self.args = args
        self.early_break = False
        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
        else:
            self.has_batch_norm = False
        with tf.device(args.device):
            self.optimizer = tf.optimizers.Adam()

    def train(self, train_data_in, val_data_in, test_data_in, max_iterations=10000,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: train data to be used.
        :param val_data: validation data to be used for early stopping. If None, train_data is splitted 
             into p_val percent for validation randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """
        # Early stopping variables
        bst_loss = np.infty

        # Main training loop
        iteration = 0
        while iteration < max_iterations:
            for train_data in train_data_in:
                self.model.input = train_data
                if self.has_batch_norm:
                    self.model.training = True
                with tf.GradientTape() as tape:
                    loss = self.model.trn_loss()
                grads = tape.gradient(loss, self.model.parms)
                grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=self.args.clip_norm) for grad in grads]
                self.optimizer.apply_gradients(zip(grads, self.model.parms))

                # Early stopping check
                iteration += 1
                self.model.training = False
                if iteration%check_every_N == 0:
                    val_data = np.concatenate([x for x in val_data_in])
                    train_data = np.concatenate([x for x in train_data_in])
                    if self.has_batch_norm:
                        self.model.update_batch_norm(train_data)
                    self.model.input = val_data
                    this_loss = self.model.trn_loss()
                    # this_loss = -np.ma.masked_invalid(sess.run(self.model.L, feed_dict={self.model.input: val_data})).mean()

                    if show_log:
                        that_loss = 0
                        test_data = np.concatenate([x for x in test_data_in])
                        self.model.input = train_data
                        train_loss = self.model.trn_loss()
                        if test_data is not None and type(test_data) == np.ndarray:
                            self.model.input = test_data
                            that_loss = self.model.trn_loss()
                            # that_loss = -np.ma.masked_invalid(sess.run(self.model.L, feed_dict={self.model.input: test_data})).mean()

                        print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}, , Test_loss: {:05.4f}".format(iteration,train_loss,this_loss, that_loss))
                    if this_loss < bst_loss:
                        bst_loss = this_loss
                        # saver.save(sess,"./"+saver_name)
                        # model_parms = sess.run(self.model.parms)
                        print("storing best weights")
                        self.model_parms = [x.numpy() for x in self.model.parms] ## unsure about efficiency here -- also based on flattening of variable list working
                        self.early_stopping_count = 0
                    else:
                        self.early_stopping_count += check_every_N
                if self.early_stopping_count >= early_stopping:
                    self.early_break = True
            if self.early_break:
                break
                
        if show_log:
            print("Training finished")
            print("Best Iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model and save batch norm mean and variance if necessary
        # saver.restore(sess,"./"+saver_name)
        print("restoring best weights")
        for m, n in zip(self.model.parms, self.model_parms):
            m.assign(n)

        if self.has_batch_norm:
            # self.model.training = False
            train_data = np.concatenate([x for x in train_data_in])
            self.model.update_batch_norm(train_data)
        
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)