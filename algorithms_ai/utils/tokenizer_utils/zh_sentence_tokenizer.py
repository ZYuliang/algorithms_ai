
import re
def _cut_paragraph_to_sentences_zh(para: str, drop_empty_line=True, strip=True, deduplicate=True):
    """
    Args:
       para: 输入文本
       drop_empty_line: 是否丢弃空行
       strip:  是否对每一句话做一次strip
       deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句

    Returns:
       sentences: list of str
    """
    if deduplicate:
        para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

    para = re.sub('([。！？\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if drop_empty_line:
        sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences
