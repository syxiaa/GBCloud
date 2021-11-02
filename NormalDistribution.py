import numpy


class NormalDistribution:
    def __init__(self, expList, varianceList):
        self.expList = expList
        self.varianceList = varianceList

    def generateRandomList(self, exp, variance, num):
        res = numpy.random.normal(exp, variance, num)
        return res

    def generateRandomPointList(self, num):
        res = []
        for i in range(len(self.expList)):
            res.append(self.generateRandomList(self.expList[i], self.varianceList[i], num))
        res = numpy.array(res)
        res = res.T
        return res


if __name__ == '__main__':
    nd = NormalDistribution([4, 3], [.9, .6])
    print(nd.generateRandomPointList(10))
