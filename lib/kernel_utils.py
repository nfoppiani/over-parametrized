import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.svm import SVC
import pandas as pd
from matplotlib import pyplot as plt

class analytic_kernel_regression(object):
    
    def __init__(self):
        self.pipeline_v = np.vectorize(self.pipeline, excluded=['self'])
        
    def compute_kernel_train_matrix(self, kernel, **kwargs_ker):
        self.kernel = kernel
        self.K_train = self.kernel(self.x_train, self.x_train, **kwargs_ker)
    
    def solve_coefficients(self, n_train):
        this_k_train = self.K_train[:n_train, :n_train]
        this_y_train = self.y_train[:n_train]
        return np.linalg.inv(this_k_train) @ this_y_train
    
    def compute_test_loss(self, coeff, loss, **kwargs_ker):
        y_test_hat = coeff @ self.kernel(self.x_train[:len(coeff)], self.x_test, **kwargs_ker)
        if loss is None:
            return y_test_hat
        else:
            if loss.__name__ == 'zero_one_loss':
                y_test_hat = np.where(y_test_hat>=0, 1, -1)
            loss_eval = loss(self.y_test, y_test_hat)
            return loss_eval
        
    def compute_train_loss(self, coeff, loss, **kwargs_ker):
        y_train_hat = coeff @ self.kernel(self.x_train[:len(coeff)], self.x_train[:len(coeff)], **kwargs_ker)
        if loss is None:
            return y_train_hat
        else:
            if loss.__name__ in ['zero_one_loss', 'hinge_loss']:
                y_train_hat = np.where(y_train_hat>=0, 1, -1)
            loss_eval = loss(self.y_train[:len(coeff)], y_train_hat)
            return loss_eval

    def pipeline(self, n_train, loss, **kwargs_ker):
        this_coeff = self.solve_coefficients(n_train)
        return (self.compute_train_loss(this_coeff, loss, **kwargs_ker), self.compute_test_loss(this_coeff, loss, **kwargs_ker))

    
class synthetic_analytic_kernel_regression(analytic_kernel_regression):
    
    def __init__(self):
        super().__init__()
    
    def sample_train_test_set(self, dimension, n_train_max, n_test, sampling='uniform', seed=0, **kwargs_gen):
        self.rng = np.random.default_rng(seed=seed)
        
        self.dimension = dimension
        self.n_train_max = n_train_max
        self.n_test = n_test
        
        if sampling == 'uniform':
            self.x_train = self.rng.uniform(size=(n_train_max, dimension), **kwargs_gen)
            self.x_test = self.rng.uniform(size=(n_test, dimension), **kwargs_gen)
        else:
            print(f'sampling {sampling} not supported yet')
            
    def compute_true_function(self, true_function, **kwargs_fn):  
        self.true_function = true_function
        self.y_train = self.true_function(self.x_train, **kwargs_fn)
        self.y_test = self.true_function(self.x_test, **kwargs_fn)
    

class mnist_analytic_kernel_regression(analytic_kernel_regression):
    
    def __init__(self, MNIST_train_path='data/mnist_train.csv', MNIST_test_path='data/mnist_test.csv', labels=(0, 1), n_train_max=1000, pre_shuffle=False):
        super().__init__()
        query = f'(label == {labels[0]}) or (label == {labels[1]})'
        MNIST_train = pd.read_csv(MNIST_train_path, sep=',').query(query)
        MNIST_test = pd.read_csv(MNIST_test_path, sep=',').query(query)
        
        self.labels = labels
        self.n_train_max = n_train_max
        
        MNIST_train.loc[MNIST_train['label'] == labels[0], 'label'] = -1
        MNIST_train.loc[MNIST_train['label'] == labels[1], 'label'] = 1
        MNIST_test.loc[MNIST_test['label'] == labels[0], 'label'] = -1
        MNIST_test.loc[MNIST_test['label'] == labels[1], 'label'] = 1

        self.dimension = MNIST_train.shape[1] - 1
        if pre_shuffle:
            MNIST_train = shuffle(MNIST_train)
        
        self.scaler = StandardScaler()
        self.scaler.fit(MNIST_train.iloc[:, 1:])
        
        self.x_train = self.scaler.transform(MNIST_train.iloc[:n_train_max, 1:])
        self.y_train = MNIST_train.iloc[:n_train_max, 0].values
        self.x_test = self.scaler.transform(MNIST_test.iloc[:, 1:])
        self.y_test = MNIST_test.iloc[:, 0].values
        

class mnist_svm_classifier(object):
    
    def __init__(self, MNIST_train_path='data/mnist_train.csv', MNIST_test_path='data/mnist_test.csv', labels=None, pre_shuffle=False):
        super().__init__()
        MNIST_train = pd.read_csv(MNIST_train_path, sep=',')
        MNIST_test = pd.read_csv(MNIST_test_path, sep=',')
        
        if labels is not None:
            query = ' or '.join([f'(label == {lab})' for lab in labels])
            MNIST_train = MNIST_train.query(query)
            MNIST_test = MNIST_test.query(query)
            
        self.labels = labels

        self.dimension = MNIST_train.shape[1] - 1
        if pre_shuffle:
            MNIST_train = shuffle(MNIST_train)
        
        self.scaler = StandardScaler()
        self.scaler.fit(MNIST_train.iloc[:, 1:])
        
        self.x_train = self.scaler.transform(MNIST_train.iloc[:, 1:])
        self.y_train = MNIST_train.iloc[:, 0].values
        self.x_test = self.scaler.transform(MNIST_test.iloc[:, 1:])
        self.y_test = MNIST_test.iloc[:, 0].values
        
        self.pipeline_v = np.vectorize(self.pipeline, excluded=['self'])
        
    def fit_model(self, n_train, **kwargs_svm):
        x_train = self.x_train[:n_train]
        y_train = self.y_train[:n_train]
        
        svm_model = SVC(**kwargs_svm)
        svm_model.fit(x_train, y_train)
        
        return svm_model
        
    def predict_and_compute_loss(self, model, x_test, y_test, loss, proba=True):
        if loss is not None:
            if loss.__name__ in ['hinge_loss', 'zero_one_loss']:
                y_test_hat = model.predict(x_test)
            elif loss.__name__ == 'mean_squared_error':
                y_test_hat = model.predict_proba(x_test)[:, 1]
            else:
                y_test_hat = model.predict_proba(x_test)
        else:
            if proba:
                y_test_hat = model.predict_proba(x_test)
            else:
                y_test_hat = model.predict(x_test)

        if loss is None:
            return y_test_hat
        else:
            loss_eval = loss(y_test, y_test_hat)
            return loss_eval
        
    def pipeline(self, n_train, loss, proba, **kwargs_svm):
        this_model = self.fit_model(n_train, **kwargs_svm)
        train_loss = self.predict_and_compute_loss(this_model, 
                                                   self.x_train[:n_train], 
                                                   self.y_train[:n_train], 
                                                   loss, 
                                                   proba)
        test_loss = self.predict_and_compute_loss(this_model, 
                                                   self.x_test, 
                                                   self.y_test, 
                                                   loss, 
                                                   proba)
        return train_loss, test_loss

    def show_image(self, i, original=False):
        this_image = self.x_train[i]
        if original:
            this_image = self.scaler.inverse_transform(this_image)
        two_d = np.reshape(this_image, (28, 28))
        plt.title(f'true label: {self.y_train[i]}')
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        plt.show()