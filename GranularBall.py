import math
import time

import numpy
import pandas
from sklearn.cluster import k_means
import matplotlib.pyplot as plt


class GranularBall:
    def __init__(self, data, NeedKurtosis=False):
        self.data = data
        # print(self.data, 'self.data')
        self.num = self.getPointNum()
        self.center, self.radius = self.getCenterAndRadius()
        self.purity, self.label = self.getPurityAndLabel()
        if NeedKurtosis and self.num > 3:
            self.Kurtosises = self.getKurtosises()

    def denoising(self):
        self.data = self.data[self.data[:, 0] == self.label]

    def getKurtosises(self):
        ks = []
        for i in range(self.data.shape[1] - 1):
            index = i + 1
            s = pandas.Series(list(self.data[:, index]))
            ks.append(s.kurt())
        return ks

    def getVarianceList(self):
        res = []
        for d in self.data.T:
            res.append(numpy.var(d))
        return res

    def getPointNum(self):
        return len(self.data)

    def getCenterAndRadius(self):
        if self.num == 1:
            return self.data[0][1:], 0
        data_no_label = self.data[:, 1:]
        center = data_no_label.mean(0)
        # print(self.data, 'center')
        radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
        return center, radius

    def getPurityAndLabel(self):
        num = self.data.shape[0]
        num_positive = sum(self.data[:, 0] == 1)
        num_negative = sum(self.data[:, 0] == 0)
        purity = max(num_positive, num_negative) / num if num else 1.0
        if num_positive >= num_negative:
            label = 1
        else:
            label = 0
        return purity, label


class GranularBallsGenerators:
    ordinal = 0

    def __init__(self, data, NeedDraw=False, NeedKurtosis=False, purity=.9, Denoising=False):
        self.ordinal += 1
        self.id = self.ordinal
        self.data = data
        self.NeedDraw = NeedDraw
        self.NeedKurtosis = NeedKurtosis
        self.purity = purity
        self.Denoising = Denoising
        # 生成粒球
        self.granularBalls = self.GenerateGranularBalls(self.data)
        # 打印kurtosis到文件
        self.printKurtosis()

    def printKurtosis(self):
        def printBlank():
            d = {
                'a': ['']
            }
            df = pandas.DataFrame(d)
            df.to_csv('Kurtosises.csv', mode='a', index=False, sep=',', header=None)

        if not self.NeedKurtosis:
            return
        dict_ = {}
        for i in range(len(self.granularBalls)):
            ball = self.granularBalls[i]
            if ball.num > 3:
                dict_['ball' + str(i)] = ball.Kurtosises
        dataFrame = pandas.DataFrame(dict_)
        printBlank()
        dataFrame.to_csv('Kurtosises.csv', mode='a', index=False, sep=',')

    def distances(self, data, p):
        return ((data - p) ** 2).sum(axis=1) ** 0.5

    def split_ball(self, granular_ball, splitting_method):
        data_no_label = granular_ball.data[:, 1:]
        if granular_ball.num == 2:
            ball1 = GranularBall(numpy.array([granular_ball.data[0]]), self.NeedKurtosis)
            ball2 = GranularBall(numpy.array([granular_ball.data[1]]), self.NeedKurtosis)
            return [ball1, ball2]
        if splitting_method == '2-means':
            label_cluster = k_means(X=data_no_label, n_clusters=2, random_state=5)[1]
            # if granular_ball.purity==.5:
            #     print(label_cluster, 'label-cluster-res')

        elif splitting_method == 'center_split':
            # 采用正、负类中心直接划分
            p_left = granular_ball.data[granular_ball.data[:, 0] == 1, 1:].mean(0)
            p_right = granular_ball.data[granular_ball.data[:, 0] == 0, 1:].mean(0)
            distances_to_p_left = self.distances(data_no_label, p_left)
            distances_to_p_right = self.distances(data_no_label, p_right)

            relative_distances = distances_to_p_left - distances_to_p_right
            label_cluster = numpy.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))

        elif splitting_method == 'center_means':
            # 采用正负类中心作为 2-means 的初始中心点
            p_left = granular_ball.data[granular_ball.data[:, 0] == 1, 1:].mean(0)
            p_right = granular_ball.data[granular_ball.data[:, 0] == 0, 1:].mean(0)
            centers = numpy.vstack([p_left, p_right])
            label_cluster = k_means(X=data_no_label, n_clusters=2, init=centers, n_init=1)[1]
        else:
            return granular_ball

        ball1 = GranularBall(granular_ball.data[label_cluster == 0, :], self.NeedKurtosis)
        ball2 = GranularBall(granular_ball.data[label_cluster == 1, :], self.NeedKurtosis)
        return [ball1, ball2]

    def isOverlap(self, granular_ball_list, i):
        ball = granular_ball_list[i]
        for index in range(len(granular_ball_list)):
            ball2 = granular_ball_list[index]
            if index != i:
                if ball.num > 3 and ball.label != ball2.label and \
                        ((ball.center - ball2.center) ** 2).sum() ** .5 < (ball.radius + ball2.radius):
                    return True
        return False

    def deOverlap(self, granular_ball_list):
        print("去重叠前球数量", len(granular_ball_list))
        while True:
            ball_number_1 = len(granular_ball_list)
            granular_ball_list = self.splits_overlap(granular_ball_list)
            ball_number_2 = len(granular_ball_list)
            if ball_number_1 == ball_number_2:
                break
        print("去重叠后球数量", len(granular_ball_list))
        return granular_ball_list

    def splits_overlap(self, granular_ball_list):
        granular_ball_list_new = []
        for i in range(len(granular_ball_list)):
            granular_ball = granular_ball_list[i]
            if self.isOverlap(granular_ball_list, i):
                granular_ball_list_new.extend(self.split_ball(granular_ball, '2-means'))
            else:
                granular_ball_list_new.append(granular_ball)
        return granular_ball_list_new

    def splits(self, granular_ball_list, purity, splitting_method):
        granular_ball_list_new = []

        for i in range(len(granular_ball_list)):
            granular_ball = granular_ball_list[i]
            p = granular_ball.purity
            if p >= purity:
                granular_ball_list_new.append(granular_ball)
            else:
                granular_ball_list_new.extend(self.split_ball(granular_ball, splitting_method))
        return granular_ball_list_new

    def GenerateGranularBalls(self, data):
        purity = self.purity
        granular_ball_list = [GranularBall(data, self.NeedKurtosis)]
        while True:
            ball_number_1 = len(granular_ball_list)
            granular_ball_list = self.splits(granular_ball_list, purity=purity, splitting_method='2-means')
            ball_number_2 = len(granular_ball_list)
            if ball_number_1 == ball_number_2:
                break

        # granular_ball_list = self.deOverlap(granular_ball_list)
        if self.NeedDraw:
            self.plot_gb(granular_ball_list, 0)
        if self.Denoising:
            for ball in granular_ball_list:
                ball.denoising()
        return granular_ball_list

    # 绘图
    def plot_gb(self, granular_ball_list, plt_type=0):
        # if not self.NeedDraw:
        #     return
        color = {0: 'r', 1: 'k'}
        plt.figure(figsize=(5, 4))
        plt.axis([0, 1.2, 0, 1])
        for granular_ball in granular_ball_list:
            label = granular_ball.label
            center = granular_ball.center
            radius = granular_ball.radius
            if plt_type == 0:
                data0 = granular_ball.data[granular_ball.data[:, 0] == 0]
                data1 = granular_ball.data[granular_ball.data[:, 0] == 1]
                plt.plot(data0[:, 1], data0[:, 2], '.', color=color[0], markersize=3)
                plt.plot(data1[:, 1], data1[:, 2], '.', color=color[1], markersize=3)
            if plt_type == 0 or plt_type == 1:
                theta = numpy.arange(0, 2 * numpy.pi, 0.01)
                x = center[0] + radius * numpy.cos(theta)
                y = center[1] + radius * numpy.sin(theta)
                plt.plot(x, y, color[label], linewidth=0.8)
            plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])
        # plt.grid(linestyle='-', color='#D3D3D3')
        plt.axis('equal')
        # plt.title('the number of GBs: {}'.format(ball_num_str))
        # plt.savefig(r'H:\mrgui\论文\Graduation thesis\03 毕业论文\图\图4.1 粒球的分裂 ' + ball_num_str + '.png')
        # plt.close()
        plt.show()


if __name__ == "__main__":
    data = pandas.read_csv("./DataSet/fourclass10_change.csv").values
    data = data[:, [0, 1, 2]]
    gbg = GranularBallsGenerators(data, NeedDraw=True, NeedKurtosis=False)
    balls = gbg.granularBalls
    # for i in range(len(balls)):
    #     print(balls[i].center, 'center')
    # print(gbg.isOverLap(balls, i))
    # for ball in balls:
    #     if ball.num > 2:
    #         print(ball.Kurtosises, '---', ball.num)
