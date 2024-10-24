def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

class NeuralNetMLP(object):
    """ 前馈神经网络 / 多层感知器分类器。

    参数
    ------------
    n_hidden : int（默认：30）
        隐藏层神经元的数量。
        （就像大脑中的神经元数量，更多的神经元可以让模型学习更复杂的模式。）
    l2 : float（默认：0.）
        L2正则化的λ值。
        （正则化可以防止模型过度拟合，类似于在考试中避免死记硬背。如果l2=0，则不进行正则化。）
    epochs : int（默认：100）
        遍历训练集的次数。
        （每次遍历都可以让模型学得更好，就像多次阅读课本可以更好地记住内容。）
    eta : float（默认：0.001）
        学习率。
        （控制模型学习的步伐，步子太大容易错过，步子太小学习太慢。）
    shuffle : bool（默认：True）
        隨機
        如果为True，每个epoch都会打乱训练数据，防止陷入循环。
        （就像洗牌，可以防止模型记住数据的顺序。）
    minibatch_size : int（默认：1）
        每个小批量中的训练样本数量。
        （一次看多少个样本，类似于一次复习多少题目。）
    seed : int（默认：None）
        用于初始化权重和打乱数据的随机种子。
        （设定随机种子可以确保每次运行结果一致，就像使用相同的起始条件。）

    属性
    -----------
    eval_ : dict
      一个字典，用于收集训练过程中每个epoch的损失、训练准确率和验证准确率。
      （可以用来观察模型的学习情况，就像记录每次考试的成绩。）

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """ 将标签编码为独热（one-hot）表示形式

        参数
        ------------
        y : array，形状 = [n_samples]
            目标值（标签），例如类别编号。

        返回
        -----------
        one-hot : array，形状 = (n_samples, n_labels)
            独热编码的数组。
            （例如，如果有3个类别，标签为1的样本将被编码为[0,1,0]。）

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """ 计算S形函数（sigmoid）

            （将输入映射到0到1之间，类似于概率。）

        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """ 计算前向传播步骤

            （将输入数据经过网络，得到输出。）

        """

        # 第一步：计算隐藏层的输入
        # 输入矩阵 [样本数, 特征数] 与权重矩阵 [特征数, 隐藏层神经元数] 相乘
        # 得到 [样本数, 隐藏层神经元数] 的矩阵
        z_h = np.dot(X, self.w_h) + self.b_h

        # 第二步：通过激活函数计算隐藏层的输出
        a_h = self._sigmoid(z_h)

        # 第三步：计算输出层的输入（从隐藏层到输出层）
        # [样本数, 隐藏层神经元数] 与权重矩阵 [隐藏层神经元数, 类别数] 相乘
        # 得到 [样本数, 类别数] 的矩阵
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # 第四步：通过激活函数计算最终输出
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """ 计算损失函数。

        参数
        ----------
        y_enc : array，形状 = (样本数, 类别数)
            独热编码的真实标签。
        output : array，形状 = [样本数, 输出神经元数]
            输出层的激活值（模型的预测结果）

        返回
        ---------
        cost : float
            加入正则化项后的总损失

        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term

        # 注意：
        # 如果激活值接近0或1，计算log(0)会导致错误（数学上未定义）。
        # 为了避免这种数值问题，可以在激活值中加入一个很小的数（如1e-5）。
        # 例如：
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        # 这样可以防止log(0)的发生。

        return cost

    def predict(self, X):
        """ 预测类别标签

        参数
        -----------
        X : array，形状 = [样本数, 特征数]
            原始特征的输入层。

        返回:
        ----------
        y_pred : array，形状 = [样本数]
            预测的类别标签。

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ 从训练数据中学习权重。

        参数
        -----------
        X_train : array，形状 = [样本数, 特征数]
            原始特征的输入层。
        y_train : array，形状 = [样本数]
            目标类别标签。
        X_valid : array，形状 = [样本数, 特征数]
            训练过程中用于验证的样本特征。
        y_valid : array，形状 = [样本数]
            训练过程中用于验证的样本标签。

        返回:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]  # 类别标签的数量
        n_features = X_train.shape[1]

        ########################
        # 权重初始化
        ########################

        # 初始化输入层到隐藏层的权重和偏置
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # 初始化隐藏层到输出层的权重和偏置
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # 用于格式化进度显示
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # 遍历训练集的epochs
        for i in range(self.epochs):

            # 迭代小批量
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # 前向传播
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # 反向传播
                ##################

                # [样本数, 类别数]
                sigma_out = a_out - y_train_enc[batch_idx]

                # [样本数, 隐藏层神经元数]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [样本数, 类别数] dot [类别数, 隐藏层神经元数]
                # -> [样本数, 隐藏层神经元数]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [特征数, 样本数] dot [样本数, 隐藏层神经元数]
                # -> [特征数, 隐藏层神经元数]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [隐藏层神经元数, 样本数] dot [样本数, 类别数]
                # -> [隐藏层神经元数, 类别数]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # 正则化和权重更新
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # 偏置项不进行正则化
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # 偏置项不进行正则化
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # 评估
            #############

            # 训练过程中每个epoch后的评估
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float64) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i + 1, self.epochs, cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

    def save_weights(self, file_path='mlp_weights.npz'):
        np.savez_compressed(file_path,
                            b_h=self.b_h,
                            w_h=self.w_h,
                            b_out=self.b_out,
                            w_out=self.w_out)
        print(f"Weights saved to {file_path}")


#-----------------------------------------------------------------
import os
import sys
import struct
import numpy as np

# 測試 MINST 載入資料
X_train, Y_train = load_mnist('', kind='train')
#print("row: %d , column: %d" % (X_train.shape[0], X_train.shape[1]))
X_test, Y_test = load_mnist('', kind='t10k')
#print("row: %d , column: %d" % (X_test.shape[0], X_test.shape[1]))

# 初始化MLP分類器
nn = NeuralNetMLP(n_hidden=300, l2=0.01, epochs=400, eta=0.001, minibatch_size=256, shuffle=True, seed=1)
print("n_hidden : " , nn.n_hidden, " l2 : ", nn.l2, " epochs : ", nn.epochs, " eta : ", nn.eta, " minibatch_size : ", nn.minibatch_size)

# 訓練MLP分類器
nn.fit(X_train=X_train[:55000],
       X_valid=X_train[55000:],
       y_train=Y_train[:55000],
       y_valid=Y_train[55000:])

nn.save_weights('mlp_weights.npz')
# 測試集評估
y_test_pred = nn.predict(X_test)
test_acc = cp.sum(Y_test == y_test_pred) / X_test.shape[0]
print(f'Test Accuracy: {test_acc * 100:.2f}%')