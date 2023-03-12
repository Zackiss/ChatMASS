import requests  # 导入requests包
from bs4 import BeautifulSoup
import re
import json


def get_article(url):
    html = requests.get(url)
    html.encoding = 'gb2312'
    soup = BeautifulSoup(html.text, 'lxml')
    father = soup.find(class_="mainNewsContent")
    herf = soup.select(".textpage")

    data = father.contents[:-1]
    for i in range(len(data)):
        data[i] = str(data[i]).replace("<p>", "").replace("</p>", "") \
            .replace("\xa0", "").replace(" ", "") \
            .replace('<aclass="UBBWordLink"href="http://www.shengyiguan.com/"target="_blank">', "") \
            .replace('</a>', "")
    data = data[4:]
    for i in range(len(data) - 1, -1, -1):
        if data[i] == '\n':
            data.remove('\n')
    return data


def get_herf(url):
    html = requests.get(url)
    html.encoding = 'gb2312'
    soup = BeautifulSoup(html.text, 'lxml')
    herf = soup.select(".textpage")

    pattern = re.compile(r'href="(.*?)">')
    herf = pattern.findall(str(herf))[:-1]
    data = get_article(url)
    for i in range(len(herf)):
        url = "http://shengyiguankb.com/SM_tuwen/" + herf[i]
        child_data = get_article(url)
        data += child_data
    return data


cur = 11
# 11, 354
while cur < 355:
    articles = []
    for i in range(0, 25):
        try:
            url_base = 'http://shengyiguankb.com/SM_tuwen/SY_{0}.html'.format(cur)
            article = get_herf(url_base)
            articles += article
            print("article {0} settled!".format(cur))
            cur += 1
        except Exception as e:
            print("no such article")
            cur += 1
    with open('art{0}.json'.format(cur), 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print("file {0} saved!".format(cur))
# with open('art{0}.json'.format(cur), 'r') as f:
#     score = json.load(f)
