import os
import sys
import struct
import cupy as cp

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = cp.fromfile(lbpath, dtype=cp.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = cp.fromfile(imgpath, dtype=cp.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

class NeuralNetMLP(object):
    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.random = cp.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        onehot = cp.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        return 1. / (1. + cp.exp(-cp.clip(z, -250, 250)))

    def _forward(self, X):
        z_h = cp.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)
        z_out = cp.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        L2_term = (self.l2 * (cp.sum(self.w_h ** 2.) + cp.sum(self.w_out ** 2.)))
        term1 = -y_enc * cp.log(output + 1e-5)
        term2 = (1. - y_enc) * cp.log(1. - output + 1e-5)
        cost = cp.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = cp.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        n_output = cp.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        self.b_h = cp.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b_out = cp.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)

        for i in range(self.epochs):
            indices = cp.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                sigma_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1. - a_h)
                sigma_h = cp.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h

                grad_w_h = cp.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = cp.sum(sigma_h, axis=0)
                grad_w_out = cp.dot(a_h.T, sigma_out)
                grad_b_out = cp.sum(sigma_out, axis=0)

                delta_w_h = grad_w_h + self.l2 * self.w_h
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((cp.sum(y_train == y_train_pred)).astype(cp.float64) / X_train.shape[0])
            valid_acc = ((cp.sum(y_valid == y_valid_pred)).astype(cp.float64) / X_valid.shape[0])

            sys.stderr.write(f'\r{i+1}/{self.epochs} | Cost: {cost:.2f} | Train/Valid Acc.: {train_acc*100:.2f}%/{valid_acc*100:.2f}% ')
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

    def save_weights(self, file_path='mlp_weights.npz'):
        cp.savez_compressed(file_path, b_h=self.b_h, w_h=self.w_h, b_out=self.b_out, w_out=self.w_out)
        print(f"Weights saved to {file_path}")

# 測試 MNIST 資料載入
X_train, Y_train = load_mnist('', kind='train')
X_test, Y_test = load_mnist('', kind='t10k')

# 初始化 MLP 分類器
nn = NeuralNetMLP(n_hidden=400, l2=0.02, epochs=1000, eta=0.0005, minibatch_size=256, shuffle=True, seed=1)
print(f"n_hidden: {nn.n_hidden}, l2: {nn.l2}, epochs: {nn.epochs}, eta: {nn.eta}, minibatch_size: {nn.minibatch_size}")

# 訓練 MLP 分類器
nn.fit(X_train=X_train[:55000], X_valid=X_train[55000:], y_train=Y_train[:55000], y_valid=Y_train[55000:])
nn.save_weights('mlp_weights.npz')

# 測試集評估
y_test_pred = nn.predict(X_test)
test_acc = cp.sum(Y_test == y_test_pred) / X_test.shape[0]
print(f'Test Accuracy: {test_acc * 100:.2f}%')
