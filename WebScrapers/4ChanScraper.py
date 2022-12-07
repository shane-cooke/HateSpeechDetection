from bs4 import BeautifulSoup
import requests
from html.parser import HTMLParser

URL = "https://boards.4chan.org/pol/"
page = requests.get(URL)

page = page.text

class Parse(HTMLParser):
    def handle_data(self, data):
        if((len(data) > 20) and (data[0].isalpha())):
            print("Comment: ", data, '\n')
        
testParser = Parse()
testParser.feed(page)