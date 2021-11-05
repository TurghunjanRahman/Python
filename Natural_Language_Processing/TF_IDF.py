"""
 -*- coding: utf-8 -*-
Author: Turghunjan Rahman
Email: turghunjanrahman@hotmail.com
Date: 11/5/2021 9:51 AM
"""

import numpy as np
import jieba

# participle
def participle(data):
    participle_list = list(jieba.lcut(par) for par in data)
    return participle_list

# TF-IDF Model
class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.tf = []
        self.idf = {}
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log(self.documents_number/(value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_docments_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list

if __name__ == '__main__':
    data = [
        '机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科',
        '专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能',
        '它是人工智能核心，是使计算机具有智能的根本途径。',
        '机器学习是人工智能及模式识别领域的共同研究热点，其理论和方法已被广泛应用于解决工程应用和科学领域的复杂问题。',
        '机器学习是研究怎样使用计算机模拟或实现人类学习活动的科学，是人工智能中最具智能特征，最前沿的研究领域之一!'
    ]
    test_text = '传统机器学习的研究方向主要包括决策树、随机森林、人工神经网络、贝叶斯学习等方面的研究'
    test_text_participle = list(jieba.lcut(test_text))
    model = TF_IDF_Model(participle(data))
    result = model.get_docments_score(test_text_participle)
    print(result)

    # Return value: [0.4853969226271291, 0.01394647195713811, 0.0, 0.022314355131420976, 0.03847302608865685]




