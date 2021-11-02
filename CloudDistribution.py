import numpy as np
from numpy import mat, zeros
from numpy.random.mtrand import uniform


class CloudDistribution:
    '''
    #功能更具输入按照云模型产生随机点
    #输入：n*m维矩阵，四种逆向云发生器的名字
    '''

    def __init__(self, ball, method='BCT_4thM'):
        # for d in data:
        # print(d, 'cloud--dis--')
        # print(data)
        self.center = ball.center
        self.radius = ball.radius
        self.data = ball.data[:, 1:]
        # print(self.data, 'cloud')
        self.concept_intension = []
        self.method = method
        if self.method == 'BCT_4thM':
            for i in range(self.data.shape[1]):
                self.concept_intension.append(self.BCT_4thM(self.data[:, i]))
        elif self.method == 'BCT':
            for i in range(self.data.shape[1]):
                self.concept_intension.append(self.BCT(self.data[:, i]))
        elif self.method == 'Second_MBCT_SD':
            for i in range(self.data.shape[1]):
                self.concept_intension.append(self.Second_MBCT_SD(self.data[:, i]))
        else:
            for i in range(self.data.shape[1]):
                self.concept_intension.append(self.Second_MBCT_SR(self.data[:, i]))

    #        print(self.concept_intension)

    def BCT(self, data):
        ex = np.mean(data)
        en = (np.pi / 2) ** 0.5 * np.mean(np.abs(data - ex))
        s2 = np.var(data, ddof=1)
        he = (np.abs(s2 - en ** 2)) ** 0.5
        return ex, en, he

    def BCT_4thM(self, data):
        ex = np.mean(data)
        M4 = np.mean((data - ex) ** 4)
        s2 = np.var(data, ddof=1)
        en = (np.abs((9 * s2 ** 2 - M4)) / 6) ** 0.25
        he = (np.abs(s2 - en ** 2)) ** 0.5
        return ex, en, he

    def Second_MBCT_SD(self, data):
        m = int((data.shape[0] ** 0.5))
        ex = np.mean(data)
        np.random.shuffle(data)
        y_2 = []
        for i in range(m):
            data_r = data[int(data.shape[0] / m * i):int((data.shape[0] / m) * (i + 1))]
            y_2.append(np.var(data_r, ddof=1))
        d_y_2 = np.var(y_2, ddof=1)
        e_y_2 = np.mean(y_2)
        en_2 = 0.5 * np.sqrt(np.abs(4 * e_y_2 * e_y_2 - 2 * d_y_2))
        he_2 = np.abs(e_y_2 - en_2)
        return ex, np.sqrt(en_2), np.sqrt(he_2)

    def Second_MBCT_SR(self, data):
        m = int(data.shape[0] ** 0.5)
        ex = np.mean(data)
        y_2 = []
        for i in range(m):
            data_r = data[np.random.randint(0, data.shape[0], (int((data.shape[0] / m) * 2),))]
            y_2.append(np.var(data_r, ddof=1))
        d_y_2 = np.var(y_2, ddof=1)
        e_y_2 = np.mean(y_2)
        en_2 = 0.5 * np.sqrt(np.abs(4 * e_y_2 * e_y_2 - 2 * d_y_2))
        he_2 = np.abs(e_y_2 - en_2)
        return ex, np.sqrt(en_2), np.sqrt(he_2)

    def Forward_cloud_model(self, ex, en, he, num):
        '''
        #正向云发生器
        '''
        X = np.random.normal(loc=en, scale=he, size=num)
        for i in range(num):
            Enn = X[i]
            X[i] = np.random.normal(loc=ex, scale=np.abs(Enn), size=1)
        return X

    def generateRandomPoint(self):
        point = np.zeros(self.data.shape[1])
        while point[0] == 0 or np.sqrt(np.sum(np.square((point - self.center)))) >= self.radius:
            for i in range(self.data.shape[1]):
                ex, en, he = self.concept_intension[i]
                Enn = np.random.normal(loc=en, scale=he, size=1)
                p = np.random.normal(loc=ex, scale=np.abs(Enn), size=1)
                point[i] = p
        return point

    def generateRandomPointList(self, num):
        """
        #输入：产生样本的个数
        #输出：num*（m-1）的矩阵，第一列的标签我不知道怎么搞
        """

        # point_list = []
        # for i in range(num):
        #     point_list.append(self.generateRandomPoint())

        point_list = np.zeros((num, self.data.shape[1]))
        for i in range(self.data.shape[1]):
            ex, en, he = self.concept_intension[i]
            point_list[:, i] = self.Forward_cloud_model(ex, en, he, num)

        return np.array(point_list)


if __name__ == "__main__":
    a = np.random.randn(5, 5)
    print(a)
    ud = CloudDistribution(a)
    print(ud.generateRandomPointList(10))
