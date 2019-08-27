#分类
##导包
from numpy import *
import re
import random
##函数以及功能说明
createVocabList():将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
setOfWords2Vec():根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
bagOfWords2VecMN():根据vocabList词汇表，构建词袋模型
trainNB0():朴素贝叶斯分类器训练函数
classifyNB():朴素贝叶斯分类器分类函数


