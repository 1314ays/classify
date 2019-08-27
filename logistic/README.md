#logistic回归
##数据准备
horseColicTest.txt
horseColicTraining.txt
##函数说明
def sigmoid(inX):sigmoid函数
def stocGradAscent1(dataMatrix,labelMat,numIter=150):改进的随机梯度上升算法
dataMatrix - 数据数组
classLabels - 数据标签
numIter - 迭代次数
def classifyVector(inX,weights):二分类问题进行分类
def colicTest():使用Python写的Logistic分类器做预测
def multiTest():调用函数colicTest()10次并求结果的平均值
