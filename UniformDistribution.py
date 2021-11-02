import numpy as np
from numpy import mat, zeros
from numpy.random.mtrand import uniform


class UniformDistribution:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.he = .1  # 百分比

    def getVariance(self):
        return (self.radius * 2) ** 2 / 12

    def getMean(self):
        return self.center

    def generateRandomPoint(self):
        # enx = [self.generateEnx(c - self.radius, c + self.radius) for c in self.center]
        return [uniform(low=self.center[index] - self.radius,
                        high=self.center[index] + self.radius)
                for index in range(len(self.center))]

    def generateRandomPointList(self, num):
        point_list = []
        for i in range(num):
            point_list.append(self.generateRandomPoint())
        return point_list

    # def generateEnx(self, a, b):
    #     return 0
        # return uniform(low=- self.he * (b - a), high=self.he * (b - a)) #超熵


if __name__ == "__main__":
    ud = UniformDistribution(np.array([1, 1, 1, 1]), np.float64(4))
    print(ud.generateRandomPointList(10))
