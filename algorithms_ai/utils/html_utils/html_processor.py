from html import unescape as html_unescape

from bs4 import BeautifulSoup


class HtmlProcessor:
    # 处理html文本，将html文本转为python字符串
    def __init__(self):
        pass

    def run(self, text):
        text = BeautifulSoup(text, "lxml").text  # 把 <>  括号中的内容去除 ，beautiful比一般的规则去除要更智能一点， 不同块之间以\n分隔
        text = html_unescape(text)  # 把转义字符还原，比如 ’&lt;‘ 还原为 ’<‘
        return text


if __name__ == '__main__':
    html_str = """<html><head><title>TheDormouse"s story</title></head><body><class="title"><b>The Dormouse"s stopy</b></p>
                  <pclass="story">Once upon a time there \n were three littlesisters;and &\t; their names  where &lt;
                  <ahref="http://example.com/elsie" class="sister"id="link1"><!--Elsie--></a>
               """
    h2 = "Oral acalabrutinib 100 mg twice daily was administered with or without intravenous pembrolizumab 200 mg on day 1 of each 3-week cycle."
    print(HtmlProcessor().run(html_str))
