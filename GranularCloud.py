import math

import numpy as np
import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier

import GBCloud.GranularBall
from GBCloud.CloudDistribution import CloudDistribution
from GBCloud.NormalDistribution import NormalDistribution
from GBCloud.UniformDistribution import UniformDistribution

from collections import Counter


def choose_lable(data):
    """找出多数类的标签"""
    tmp = 0
    label = 0
    for key, value in Counter(data[:, 0]).items():
        if value > tmp:
            label = key
            tmp = value
    return label


def right_rate_min(alist, blist, lable):  # alist测试集真实标签 blist模型预测标签  lable多数类标签
    """ 预测的标签中 计算少数类中的正确率有多少  """
    n = len(set(alist))
    cata_list = [[] for _ in range(n - 1)]  # 分类  每个类在一个[]中
    right_list = [[] for _ in range(n - 1)]  # 每种类分对了的 放在一个[] 中
    index_list = 0
    sum = 0
    for i in range(len(alist)):
        flag = 1
        if alist[i] == lable:
            continue
        else:
            if alist[i] == blist[i]:
                for j in range(n - 1):
                    if (right_list[j]) and alist[i] == right_list[j][0]:
                        right_list[j].append(alist[i])
                        flag = 0
                        break
                if flag:
                    right_list[index_list].append(alist[i])
                    index_list += 1

    for i in range(len(alist)):
        if alist[i] == lable:
            continue
        else:
            for j in range(n - 1):
                if (right_list[j]) and alist[i] == right_list[j][0]:
                    cata_list[j].append(alist[i])

    for i in range(n - 1):
        if right_list[i] and cata_list[i]:
            sum += (len(right_list[i]) / len(cata_list[i])) / (n - 1)
    return sum


def __get_f_measure_g_mean(real_label, predict_label):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(real_label)):
        if predict_label[i] == 1:  # 少数类
            if real_label[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if real_label[i] == 1:
                fn += 1
            else:
                tn += 1
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)
    pp_value = tp / (tp + fp) if (tp + fp) else 0

    recall = tp_rate
    precision = pp_value

    f_measure = (2 * recall * precision) / (recall + precision) if pp_value else 0
    g_mean = math.sqrt(tp_rate * tn_rate)
    return f_measure, g_mean, (tp, fp, fn, tn)


def get_metrics(data_train, data_test, clf=KNeighborsClassifier()):
    if len(Counter(data_train[:, 0])) < 2:
        metrics = {'accuracy': -1,
                   'precision': -1,
                   'recall': -1,
                   'auc': -1,
                   'f1': -1,
                   'g_mean': -1,
                   'tfpn': (-1, -1, -1, -1),
                   'right_rate_min_score': -1}
        print('{} Only one class in the data set {}'.format('  ' * 10, '!' * 10))
        return metrics

    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, roc_curve, precision_score

    if len(Counter(data_train[:, 0])) == 2:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1]
        accuracy = accuracy_score(data_test[:, 0], predict)  # 准确率
        precision = precision_score(data_test[:, 0], predict)
        recall = recall_score(data_test[:, 0], predict)
        auc = roc_auc_score(data_test[:, 0], predict_proba)  # roc曲线面积
        f1 = f1_score(data_test[:, 0], predict)
        g_mean, tfpn = __get_f_measure_g_mean(data_test[:, 0], predict)[1:]
        fprs, tprs, thresholds = roc_curve(data_test[:, 0], predict_proba)

        right_rate_min_score = right_rate_min(data_test[:, 0], predict, choose_lable(data_train))

        # plt.plot(fprs, tprs, 'slategray')

        metrics = {'accuracy': accuracy,
                   'precision': precision,
                   'recall': recall,
                   'auc': auc,
                   'f1': f1,
                   'g_mean': g_mean,
                   'tfpn': tfpn,
                   'right_rate_min_score': right_rate_min_score
                   }
    else:
        clf.fit(data_train[:, 1:], data_train[:, 0])
        predict = clf.predict(data_test[:, 1:])
        predict_proba = clf.predict_proba(data_test[:, 1:])[:, 1]
        accuracy = accuracy_score(data_test[:, 0], predict)

        # plt.plot(fprs, tprs, 'slategray')

        metrics = {'accuracy': accuracy}

    return metrics


class GranularCloud:
    def __init__(self, NeedDraw=False, NeedKurtosis=False, Distribution='uniform', FindBestP=False, Denoising=False,
                 NeedKnnValidate=False):
        self.result_data = None
        self.Distribution = Distribution
        self.granular_balls = []
        self.NeedDraw = NeedDraw  # 是否绘图
        self.NeedKurtosis = NeedKurtosis  # 是否计算峰度
        self.FindBestP = FindBestP  # 是否纯度寻优
        self.Denoising = Denoising  # 是否去噪
        self.plot_gb = False  # 粒球类内部是否绘图
        self.NeedKnnValidate = NeedKnnValidate

        self.data_origin = None
        self.data_train = None
        self.data_validate = None  # 验证集
        self.gbg = None  # 粒球生成器

    def OverSampling(self):
        from sklearn.neighbors import KNeighborsClassifier
        n0 = 0
        n1 = 0
        for granular_ball in self.granular_balls:
            if granular_ball.label == 0:
                n0 += granular_ball.num
            else:
                n1 += granular_ball.num
        if n0 > n1:
            granular_balls_min = \
                [ball for ball in self.granular_balls if ball.label == 1]
        else:
            granular_balls_min = \
                [ball for ball in self.granular_balls if ball.label == 0]
        if n0 < n1:
            granular_balls_max = \
                [ball for ball in self.granular_balls if ball.label == 1]
        else:
            granular_balls_max = \
                [ball for ball in self.granular_balls if ball.label == 0]
        AllNumInsert = abs(n0 - n1)
        SumRadius = sum([ball.radius * ball.radius for ball in granular_balls_min])

        def getArray(num):
            if n0 > n1:
                return numpy.ones(num)
            else:
                return numpy.zeros(num)

        self.draw_balls(self.granular_balls)
        for ball in granular_balls_min:
            if ball.num > 3:
                r = ball.radius
                numInsert = int(AllNumInsert * r * r / SumRadius) + 1

                if self.Distribution == 'uniform':
                    ud = UniformDistribution(ball.center, r)
                    psInsert = numpy.array(
                        ud.generateRandomPointList(numInsert))
                elif self.Distribution == 'normal':
                    nd = NormalDistribution(ball.center, ball.getVarianceList())
                    psInsert = numpy.array(
                        nd.generateRandomPointList(numInsert))
                elif self.Distribution == 'cloud':
                    cd = CloudDistribution(ball)
                    psInsert = numpy.array(cd.generateRandomPointList(numInsert))

                #
                # if ball.num > 3 and numpy.mean(ball.Kurtosises) < 0:
                #     ud = UniformDistribution(ball.center, r)
                #     psInsert = numpy.array(
                #         ud.generateRandomPointList(numInsert))
                # elif ball.num > 3 and numpy.mean(ball.Kurtosises) > 0:
                #     cd = CloudDistribution(ball.data)
                #     psInsert = numpy.array(cd.generateRandomPointList(numInsert))

                # elif self.Distribution == 'cloud':
                #     nd = NormalDistribution(ball.center, ball.getVarianceList())
                #     psInsert = numpy.array(nd.generateRandomPointList(numInsert))

                # 添加标签
                try:
                    psInsert = numpy.insert(psInsert, 0, getArray(psInsert.shape[0]), axis=1)
                except numpy.AxisError:
                    print('oversampling*************')
                    # print(psInsert.shape, numInsert)
                psInsert_final = []
                if self.NeedKnnValidate:
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(self.result_data[:, 1:], self.result_data[:, 0])
                    for point in psInsert:
                        predict = knn.predict([point[1:]])
                        if predict[0] == ball.label:
                            psInsert_final.append(point)
                else:
                    psInsert_final = psInsert
                if len(psInsert_final) > 0:
                    ball.data = numpy.append(ball.data, numpy.array(psInsert_final), axis=0)
        self.draw_balls(self.granular_balls)
        self.granular_balls = granular_balls_max + granular_balls_min

    def draw_balls(self, balls):
        if self.NeedDraw:
            self.gbg.plot_gb(balls, plt_type=0)

    def main(self):
        data_positive = self.data_origin[self.data_origin[:, 0] == 1, :]
        data_negative = self.data_origin[self.data_origin[:, 0] == 0, :]
        lenOfp = len(data_positive)
        lenOfN = len(data_negative)
        self.data_train = np.empty(shape=[0, len(self.data_origin[0])])
        self.data_train = np.append(self.data_train, data_positive[0:int(lenOfp * .8), :], axis=0)
        self.data_train = np.append(self.data_train, data_negative[0:int(lenOfN * .8), :], axis=0)
        self.data_validate = np.empty(shape=[0, len(self.data_origin[0])])
        self.data_validate = np.append(self.data_validate, data_positive[int(lenOfp * .8):, :], axis=0)
        self.data_validate = np.append(self.data_validate, data_negative[int(lenOfN * .8):, :], axis=0)

        def get_result_from(purity):
            self.gbg = GBCloud.GranularBall.GranularBallsGenerators(self.data_train, NeedDraw=self.plot_gb,
                                                                    NeedKurtosis=self.NeedKurtosis,
                                                                    purity=purity, Denoising=self.Denoising)
            self.granular_balls = self.gbg.granularBalls
            self.result_data = np.empty(shape=[0, len(self.granular_balls[0].data[0])])
            for ball in self.granular_balls:
                try:
                    self.result_data = np.append(self.result_data, ball.data, axis=0)
                except ValueError:
                    print('*************')
                    print(self.result_data)
                    print(ball.data_train)
                    print('*************')
            # print(self.result_data, self.granular_balls[0].data,'self.result_data')
            self.OverSampling()
            result_data = np.empty(shape=[0, len(self.granular_balls[0].data[0])])
            for ball in self.granular_balls:
                try:
                    result_data = np.append(result_data, ball.data, axis=0)
                except ValueError:
                    print('*************')
                    print(result_data)
                    print(ball.data_train)
                    print('*************')

            def getM(metrics):
                res = 0
                for key in metrics:
                    if key != 'tfpn':
                        res += metrics[key]
                return res

            metrics = get_metrics(result_data, self.data_validate)
            return getM(metrics), result_data

        i = .6
        result_data_final = None
        if self.FindBestP:
            max_ = -1
            while i <= 1:
                metric, result = get_result_from(i)
                if max_ <= metric:
                    max_ = metric
                    result_data_final = result
                i += .01
        else:
            mtric, result_data_final = get_result_from(.9)
        return result_data_final[:, 1:], result_data_final[:, 0]

    def fit_resample(self, X_data_train, Y_data_train):
        print("fit_sample")
        self.data_origin = np.column_stack((Y_data_train, X_data_train))
        return self.main()

    def sample(self, X_data_train, Y_data_train):
        print("sample")
        self.data_origin = np.column_stack((Y_data_train, X_data_train))
        return self.main()

    def test(self, data):
        self.data_origin = data
        return self.main()


if __name__ == "__main__":
    data = pandas.read_csv("./DataSet/fourclass10_change.csv").values
    data = data[:, [0, 1, 2]]
    print(data.shape)

    gc = GranularCloud(NeedKurtosis=False, Distribution='cloud', NeedDraw=True)
    res = gc.test(data)
    print(res[0].shape, res[1].shape)

    gc = GranularCloud(NeedKurtosis=False, Distribution='uniform', NeedDraw=True)
    res = gc.test(data)
    print(res[0].shape, res[1].shape)

    gc = GranularCloud(NeedKurtosis=False, Distribution='normal', NeedDraw=True)
    res = gc.test(data)
    print(res[0].shape, res[1].shape)
