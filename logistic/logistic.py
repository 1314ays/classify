from numpy import *

def sigmoid(inX):
	return 1.0/(1+exp(-inX))


def stocGradAscent1(dataMatrix,labelMat,numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	for i in range(0,numIter):
	    dataIndex = range(m)
	    for j in range(0,m):
	        alpha = 4/(1.0+j+i)+0.01
	        randIndex = int(random.uniform(0,len(dataIndex)))
	        h = sigmoid(sum(dataMatrix[randIndex] * weights))
	        error = labelMat[randIndex] - h
	        weights = weights + alpha * error * dataMatrix[randIndex]
	        del(dataIndex[randIndex])
	
	return weights       
	
	
	#二分类问题进行分类
def classifyVector(inX,weights):
	prob = sigmoid(sum(inX * weights))
	if prob>0.5:
	    return 1.0
	else:
	    return 0.0
	
	#训练和测试
def colicTest():
	frTrain = open('C:\\Users\\user\Desktop\horseColicTraining.txt'); frTest = open('C:\\Users\\user\Desktop\horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	    #训练回归模型
	for line in frTrain.readlines():
	    currLine = line.strip().split('\t')
	    lineArr =[]
	    for i in range(21):
	        lineArr.append(float(currLine[i]))
	    trainingSet.append(lineArr)
	    trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
	errorCount = 0; numTestVec = 0.0
	    #测试回归模型
	for line in frTest.readlines():
	    numTestVec += 1.0
	    currLine = line.strip().split('\t')
	    lineArr =[]
	    for i in range(21):
	        lineArr.append(float(currLine[i]))
	    if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
	        errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print ("the error rate of this test is: %f" % errorRate)
	return errorRate
	
def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
	    errorSum = errorSum + colicTest()
	print ("after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))
	
if __name__=="__main__":
	multiTest()  
