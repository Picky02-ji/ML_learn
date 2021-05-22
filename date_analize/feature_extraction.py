from sklearn.feature_extraction import DictVectorizer  # 字典提取
from sklearn.feature_extraction.text import CountVectorizer  # 直接提取次数
from sklearn.feature_extraction.text import TfidfVectorizer  # 依次数提取

import jieba  # 中文分词库

#  结巴分词
def words_cut(text):
    return " ".join(list(jieba.cut(text)))  # 转为字符串再转为列表再转为字符串

# 字典特征抽取
def get_feature():
    date = [{'city': '北京', 'tem': 100},
          {'city': '上海', 'tem': 20},
          {'city': '深圳', 'tem': 40}]
    transform = DictVectorizer(sparse=False)  # 非稀疏矩阵输出
    date_now = transform.fit_transform(date)
    print('date_new:\n', date_now)
    print("date_name:\n", transform.get_feature_names())

# 文本特征抽取
def get_feeling():
    date1 = ["life is short,I like python", "life is too long,I don't like python"]
    transform = CountVectorizer(stop_words=["is", "too"])
    date_new = transform.fit_transform(date1)
    print('date_new:\n', date_new.toarray())
    print("date_name:\n", transform.get_feature_names())

# 中文提取
def count_Chinese():
    date_new = []
    for sent in date:
        date_new.append(words_cut(sent))
    transfrom = CountVectorizer()
    transfrom.fit_transform(date_new)
    print('chinses_date:\n', date_new)
    print("date_name:\n", transfrom.get_feature_names())

if __name__ == '__main__':
    date = ["今天天气真不错，我高兴极了", "今天妈妈打我了，我很伤心", "今天是愚人节，就算不开心也要装作开心的样子"]
    # get_feature()
    # get_feeling()
    count_Chinese()
