from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import urllib.parse


class DeepL:
    def __init__(self):
        options = Options()
        options.add_argument('--headless')

        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    def translate(self, from_lang: str, to_lang: str, from_text: str) -> str:
        sleep_time = 1
        from_text = urllib.parse.quote(from_text)
        url = 'https://www.deepl.com/translator#' \
              + from_lang + '/' + to_lang + '/' + from_text
        self.driver.get(url)
        self.driver.implicitly_wait(10)
        to_text = None
        for i in range(30):
            time.sleep(sleep_time)
            html = self.driver.page_source
            to_text = self.get_text_from_page_source(html)

            if to_text:
                break
        return to_text

    def get_text_from_page_source(self, html: str) -> str:
        soup = BeautifulSoup(html, features='html.parser')
        target_elem = soup.find(class_="lmt__translations_as_text__text_btn")
        text = None
        if target_elem is not None:
            text = target_elem.text
        return text


if __name__ == '__main__':
    content = """
    We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
    """
    res = DeepL().translate('en', 'zh', content)
    print(res)
