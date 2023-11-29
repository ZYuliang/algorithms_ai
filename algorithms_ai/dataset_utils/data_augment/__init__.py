"""
在原始文本上做同义替换，再输入模型转为token
nlp文本的数据增强
1. 回译 ，需要模型
2. 同义短语替换 ，需要分类，结合词的重要性
3. 数据生成
4. 同义句挖掘

输入：原始文本，以及一些相应的参数
输出：替换后的文本

"""
class BaseDataAugmentation:
    def __init__(self,**kwargs):
        pass

    def run(self, text:str,label):
        pass


class BackTranslate(BaseDataAugmentation):
    def __init__(self,):
        super(BackTranslate, self).__init__()








if __name__ == '__main__':

    t = "ACROBiosystems百普赛斯与和元生物达成战略合作，加速CGT与神经领域生物药研发上市进程"
