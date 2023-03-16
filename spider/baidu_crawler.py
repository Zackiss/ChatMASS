import time
import requests
import lxml.etree
from bs4 import BeautifulSoup


class Spider(object):
    """"""
    def __init__(self, name):
        self.session = None
        self.max_page = 1
        self.url = f'https://tieba.baidu.com/f?kw={name}'
        self.headers = {
            # 'User-Agent': '/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'
            # 'User-Agent': 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1) '
        }
        self.global_data_got = [
            {
                "query": "how's the day?",
                "answer": "not that good."
            }
        ]

    def get_frontpage(self, url):
        """"""
        response = requests.get(url, headers=self.headers)
        with open(f"tmp/{url.split('/')[-1].replace('?', '.')}.html", 'wb') as f:
            f.write(response.content)
        return response.content

    def get_responses_in_post(self, url) -> str:
        # time.sleep(random.randint(5, 10))
        response = requests.get(url, headers=self.headers)
        return response.text

    def parse_data(self, data):
        data = data.decode().replace("<!--", "").replace("-->", "")
        el_html = lxml.etree.HTML(data)
        el_list = el_html.xpath('//*[@id="thread_list"]/li/div')
        if not len(el_list):
            print("session being blocked, please change the proxy.")
            exit()
        frontpage_list = []
        for el in el_list:
            tmp = {'title': el.xpath('./div[2]/div[1]/div[1]/a/text()')[0],
                   'href': 'http://tieba.com' + el.xpath('./div[2]/div[1]/div[1]/a/@href')[0]}
            frontpage_list.append(tmp)
        # front_page: list(dict)
        for page_data in frontpage_list:
            print("----")
            print("标题: " + page_data['title'])
            herf = page_data['href']
            soup = BeautifulSoup(self.get_responses_in_post(url=herf), 'lxml')
            floors = soup.find_all('div', class_="d_post_content_main")
            for floor in floors:
                comments = floor.findChildren('div', class_="d_post_content j_d_post_content")
                for comment in comments:
                    if comment.text.replace(" ", ""):
                        print("跟帖: " + comment.text.replace(" ", ""))
        try:
            # next_url = 'https' + el_html.xpath('//a[@class="next pagination-item "]/@href')
            next_url = 'https:' + el_html.xpath('//a[contains(text(),"下一页")]/@href')[0]
        except Exception as e:
            next_url = None
        return frontpage_list, next_url

    def save_data(self, data_list):
        for post in data_list:
            print(post)

    def run(self):
        next_url = self.url
        cur_page = 0
        while cur_page < self.max_page:
            cur_page += 1
            data = self.get_frontpage(next_url)
            data_list, next_url = self.parse_data(data)
            self.save_data(data_list)
            if next_url is None:
                break
            time.sleep(30)


if __name__ == '__main__':
    spider = Spider('核战避难所')
    spider.run()
