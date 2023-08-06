
#______________________________________PEP8____________________________________
#_______________________________________________________________________
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score


class DShap(object):
    """DShap是一个用于模型训练和评估的类

    Args:
        X: 训练数据的特征
        y: 训练数据的标签
        X_test: 测试数据的特征
        y_test: 测试数据的标签
        sources: 分配给每个数据点的组的数组或字典。如果为空，则每个点都有其独立的值
        sample_weight: 在损失函数中训练样本的权重（对于支持加权训练方法的模型）
        num_test: 用于评估度量的数据点数量
        directory: 用于保存结果和图形的目录
        problem: 问题类型，'分类'或'回归'（尚未实现）
        model_family: 学习算法使用的模型家族
        metric: 评估度量
        seed: 随机种子。在并行运行蒙特卡洛样本时，我们为每个样本初始化不同的种子，以防止得到相同的置换
        overwrite: 删除现有数据并从头开始计算
        **kwargs: 模型的参数
    """

    def __init__(self, X, y, X_test, y_test, num_test, sources=None,
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """构造函数"""

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_random_seed(seed)

        # 初始化一些基本的属性
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])

        # 如果选择的模型家族是逻辑回归，则隐藏单元为空
        if self.model_family is 'logistic':
            self.hidden_units = []

        # 如果提供了目录，则设置目录并可能删除现有的文件
        if self.directory is not None:
            if overwrite and os.path.exists(directory):
                tf.gfile.DeleteRecursively(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test,
                                      sources, sample_weight)

        # 对于多分类问题，检查度量的有效性
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'

        # 判断问题是否是回归问题
        is_regression = (np.mean(self.y // 1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)

        # 如果是回归问题，则发出警告
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")

        # 返回对应的模型
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)

    # 初始化一个实例并加载或创建数据集
    def _initialize_instance(self, X, y, X_test, y_test, num_test,
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""
        # 检查sources是否为None，如果是，则将sources初始化为一个字典，每个键值对的键为X的索引，
        # 值为一个只包含该索引的数组。如果sources不为None且不是字典类型，
        # 则将sources转换为字典，键为sources中的元素，值为sources中元素等于键的索引构成的数组
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        # 函数会检查数据目录下是否存在数据.pkl文件，如果存在，则调用_load_dataset函数加载数据；
        # 如果不存在，则创建一个新的数据集，并将数据保存在数据.pkl文件中。最后，函数会创建结果的占位符。
        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            self._load_dataset(data_dir)
        else:
            self.X_heldout = X_test[:-num_test]
            self.y_heldout = y_test[:-num_test]
            self.X_test = X_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.X, self.y, self.sources = X, y, sources
            self.sample_weight = sample_weight
            data_dic = {'X': self.X, 'y': self.y, 'X_test': self.X_test,
                     'y_test': self.y_test, 'X_heldout': self.X_heldout,
                     'y_heldout':self.y_heldout, 'sources': self.sources}
            if sample_weight is not None:
                data_dic['sample_weight'] = sample_weight
                warnings.warn("Sample weight not implemented for G-Shapley")
            pkl.dump(data_dic, open(data_dir, 'wb'))
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        n_points = len(self.X)
        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number,
            n_points, n_sources, self.model_family)

     # 这个函数用来创建结果的占位符
    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                   n_points, n_sources, model_family):
        # tmc_dir和g_dir，然后创建了两个0数组，分别为mem_tmc和mem_g，并且根据模型的类型，决定是否需要存储mem_g
        tmc_dir = os.path.join(
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory, 
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        if model_family not in ['logistic', 'NN']:
            return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))

    # 加载已存在的数据集
    def _load_dataset(self, data_dir):
        '''Load the different sets of data if already exists.'''
        data_dic = pkl.load(open(data_dir, 'rb'))
        self.X_heldout = data_dic['X_heldout']
        self.y_heldout = data_dic['y_heldout']
        self.X_test = data_dic['X_test']
        self.y_test = data_dic['y_test']
        self.X = data_dic['X'] 
        self.y = data_dic['y']
        self.sources = data_dic['sources']
        if 'sample_weight' in data_dic.keys():
            self.sample_weight = data_dic['sample_weight']
        else:
            self.sample_weight = None

    # 避免并行运行时的冲突

    def _which_parallel(self, directory):
    # 获取目录下的所有文件名，
    # 然后从文件名中找出含有'mem_tmc'和'mem_g'的文件，
    # 提取文件名中的数字，并找出最大的数字加1，作为新的TMC的数量和G的数量。
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]        
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0' 
        return tmc_number, g_number

    # 参数是一个指标 根据指标的不同，有不同的计算方式
    def init_score(self, metric):
        # 例如，如果指标是准确率（accuracy），则返回测试集标签分布中数量最多的类别的频率；
        # 如果是F1分数，则计算1000次随机排列测试集标签后的F1分数的平均值；
        # 如果是AUC，则返回0.5。如果以上都不满足，则会通过模型拟合随机排列的标签计算指标，取100次计算结果的平均值。
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y,
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)

    def value(self, model, metric=None, X=None, y=None):
        # 函数根据给定的指标进行不同的评价，如准确率，F1分数，AUC等。如果指标是函数，则直接调用函数进行评估。
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data
                different from test set.
            y: Labels, if valuation is performed on a data
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')

    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True):
        # 检查是否需要计算并保存留一法的分数，然后计算G-Shapley值和TMC-Shapley值，最后保存结果。
        """Calculates data sources(points) values.

        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
        print('LOO values calculated!')
        tmc_run = True
        g_run = g_run and self.model_family in ['logistic', 'NN']
        while tmc_run or g_run:
            if g_run:
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every,
                        tolerance=tolerance,
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
            if self.directory is not None:
                self.save_results()

    # 保存迄今为止计算的结果
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory,
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory,
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g},
                 open(g_dir, 'wb'))

    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        # 检查数据源是否为None，如果是，则将其设置为一个字典，键为X的索引，值为一个只包含该索引的数组。
        # 然后，运行一定次数的迭代，每次迭代调用one_iteration函数，并将结果保存到mem_tmc和idxs_tmc中。
        """Runs TMC-Shapley algorithm.
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.one_iteration(
                tolerance=tolerance,
                sources=sources
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc,
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc,
                np.reshape(idxs, (1,-1))
            ])

    def _tol_mean_score(self):
        # 这个函数通过bagging来计算平均性能及其误差。它首先初始化一个空的得分列表，然后重启模型。
        # 如果没有设置样本权重，它会使用特征和目标变量对模型进行拟合；如果设置了样本权重，拟合过程将考虑这些权重。
        # 然后，它会循环100次，每次从测试目标变量中随机抽取与测试目标变量长度相同的样本，并计算这些样本的模型性能。
        # 最后，它计算得分的标准差和平均值，并将这两个值分别赋予tol和mean_score。
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            if self.sample_weight is None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(self.X, self.y,
                              sample_weight=self.sample_weight)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model, 
                    metric=self.metric,
                    X=self.X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        # 首先检查是否给出了数据源，如果没有，则将每个样本作为其自身的数据源；
        # 如果给出了数据源但不是字典形式的，则根据给出的数据源重新构造数据源字典。
        # 然后，它随机排序数据源，并初始化边际贡献向量、特征和目标的批处理向量、样本权重的批处理向量和截断计数器。
        # 接下来，它遍历所有数据源，每次都添加一个数据源到批处理向量中，然后拟合模型并计算新的性能得分。
        # 它还计算每个数据源的平均边际贡献，并检查新的性能得分与平均性能得分的距离是否小于等于平均性能得分的容忍度乘以平均性能得分。
        # 如果是，增加截断计数器；如果截断计数器超过5，停止迭代。
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
                    self.restart_model()
                    if sample_weight_batch is None:
                        self.model.fit(X_batch, y_batch)
                    else:
                        self.model.fit(
                            X_batch, 
                            y_batch,
                            sample_weight = sample_weight_batch
                        )
                    new_score = self.value(self.model, metric=self.metric)       
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        # 尝试克隆模型。如果无法克隆，那么它将用空的特征矩阵和目标变量来拟合模型。
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
    def _one_step_lr(self):
        # 计算G-Shapley算法的最佳学习率。它遍历一系列的学习率值，对于每个值，都使用该学习率创建并拟合一个模型，然后计算模型在测试集上的准确率。
        # 它选取使得准确率减去准确率的标准差最大的学习率。
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', 
                warm_start=False, address=address, 
                hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, self.X.shape[-1])), self.y)
                model.fit(self.X, self.y)
                accs.append(model.score(self.X_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        # 检查是否给出了数据源，如果没有，则将每个样本作为其自身的数据源；
        # 如果给出了数据源但不是字典形式的，则根据给出的数据源重新构造数据源字典。
        # 然后，它检查是否给出了学习率，如果没有，则尝试使用已经计算出的最佳学习率，如果没有计算出最佳学习率，则计算一个。
        # 接着，它创建并拟合一个模型，然后在一定的迭代次数内，对每个数据源，都计算其边际贡献，并将边际贡献加入到贡献记忆中。
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(
                    iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, 
                      sources=sources, metric=self.metric, 
                      max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.X))
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        # 留一法
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        if self.sample_weight is None:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y,
                          sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in sources.keys():
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            if self.sample_weight is not None:
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            if self.sample_weight is None:
                self.model.fit(X_batch, y_batch)
            else:
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)
                
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo
    
    def _merge_parallel_results(self, key, max_samples=None):
        # 合并并行运算的结果。在并行处理大量数据时，结果通常会分散在多个子进程中，这个函数就是为了将这些结果合并在一起。
        # 函数会从指定的目录中读取所有相关的文件，然后合并这些数据，最后将合并后的结果保存在一个新的文件中。
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) 
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.X.shape[0]))
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.X))
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory, 
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory, 
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self, max_samples=None):
        # 合并运算结果
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        if self.model_family not in ['logistic', 'NN']:
            return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    
    def performance_plots(self, vals, name=None, 
                          num_plot_markers=20, sources=None):
        # 展示画图
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 25
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                  for val in vals]
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0, 
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys())//num_plot_markers, 1)
        )
        perfs = [self._portion_performance(
            np.argsort(vals_source)[::-1], plot_points, sources=sources)
                 for vals_source in vals_sources]
        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])[::-1]),
            plot_points, sources=sources) for _ in range(10)], 0)
        plt.plot(plot_points/len(self.X) * 100, perfs[0] * 100, 
                 '-', lw=5, ms=10, color='b')
        if len(vals)==3:
            plt.plot(plot_points/len(self.X) * 100, perfs[1] * 100, 
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals)==2:
            legends = ['TMC-Shapley ', 'LOO', 'Random']
        else:
            legends = ['TMC-Shapley ', 'Random']
        plt.plot(plot_points/len(self.X) * 100, perfs[-1] * 100, 
                 '-.', lw=5, ms=10, color='g')
        plt.plot(plot_points/len(self.X) * 100, rnd * 100, 
                 ':', lw=5, ms=10, color='r')    
        plt.legend(legends)
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
            
    def _portion_performance(self, idxs, plot_points, sources=None):
        # 逐渐移除数据点的过程中，模型的性能如何变化。函数会接收一个按照数据点重要性排序的索引列表，
        # 然后从最重要的数据点开始逐个移除，每移除一个数据点就重新训练模型，并且计算新模型的性能。
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            keep_idxs = np.concatenate([sources[idx] for idx 
                                        in idxs[plot_points[i-1]:]], -1)
            X_batch, y_batch = self.X[keep_idxs], self.y[keep_idxs]
            if self.sample_weight is not None:
                sample_weight_batch = self.sample_weight[keep_idxs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))):
                    self.restart_model()
                    if self.sample_weight is None:
                        self.model.fit(X_batch, y_batch)
                    else:
                        self.model.fit(X_batch, y_batch,
                                      sample_weight=sample_weight_batch)
                    scores.append(self.value(
                        self.model,
                        metric=self.metric,
                        X=self.X_heldout,
                        y=self.y_heldout
                    ))
                else:
                    scores.append(init_score)
        return np.array(scores)[::-1]
