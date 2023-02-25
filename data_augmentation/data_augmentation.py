import random
import torch
from transformers import BertTokenizer, BertForMaskedLM
from utils import get_device, punctuation
import jieba


class EDA:
    """
    替换同意词、插入同意词、交换词的顺序、删除词
    """

    def __init__(self):
        import synonyms
        self.synonyms = synonyms
        self.stop_words = synonyms.synonyms._stopwords
        self.word_dic = {}

    def augmentation(self, text, rate=0.3):

        replace_text = self.replace(text, rate=rate)
        insert_text = self.insert(text, rate=rate)
        swap_text = self.swap(text)
        delete_text = self.delete(text, rate=rate)
        all_text = list({text, replace_text, insert_text, swap_text, delete_text})
        return all_text

    def replace(self, text, rate=0.2):
        segment = list(jieba.cut(text))
        words_index = []
        for i, s in enumerate(segment):
            if s not in self.stop_words:
                words_index.append(i)
        if not words_index:
            return text
        num = max(1, round(len(words_index) * rate))
        index = [random.choice(words_index) for _ in range(num)]

        for i in index:
            try:
                if segment[i] in self.word_dic.keys():
                    segment[i] = self.word_dic[segment[i]]
                else:
                    new_word = self.synonyms.nearby(segment[i])[0][1]
                    segment[i] = new_word
            except:
                pass

        return ''.join(segment)

    def insert(self, text, rate=0.2):
        segment = list(jieba.cut(text))
        words_index = []
        for i, s in enumerate(segment):
            if s not in self.stop_words:
                words_index.append(i)
        if not words_index:
            return text
        num = max(1, round(len(words_index) * rate))
        index = [random.choice(words_index) for _ in range(num)]

        for i in index:
            try:
                if segment[i] in self.word_dic.keys():
                    segment[i] = self.word_dic[segment[i]]
                else:
                    new_word = self.synonyms.nearby(segment[i])[0][1]
                    segment[i] = new_word
            except:
                pass

        return ''.join(segment)

    def swap(self, text):
        segment = list(jieba.cut(text))
        if len(segment) <= 2:
            return text

        choice_word = [1, 1]
        while choice_word[0] == choice_word[1]:
            choice_word = random.choices(segment, k=2)

        segment[segment.index(choice_word[0])] = choice_word[1]
        segment[segment.index(choice_word[1])] = choice_word[0]

        return ''.join(segment)

    def delete(self, text, rate=0.2):
        segment = list(jieba.cut(text))
        for i in range(len(segment)):
            if random.random() < rate:
                segment[i] = ''
        return ''.join(segment)


class AEDA:
    """
    随机添加标点
    https://arxiv.org/pdf/2108.13230.pdf
    """

    def __init__(self):
        self.punctuation = punctuation()

    def augmentation(self, text):
        length = int(len(text) * 0.3)
        if length < 2:
            return text
        punc_len = random.randint(1, length)
        puncs = random.choices(self.punctuation, k=punc_len)
        text = list(text)
        for p in puncs:
            text.insert(random.randint(0, len(text) - 1), p)
        return ''.join(text)


class BackTranslation:
    """
    回译
    """

    def __init__(self):
        from deepl import DeepL
        self.deep = DeepL()

    def augmentation(self, text):
        english = self.deep.translate('zh', 'en', text)
        translate = self.deep.translate('en', 'zh', english)
        return translate


class WoTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for word in self.pre_tokenizer(text):
            if word in self.vocab:
                split_tokens.append(word)
            else:
                split_tokens.extend(super()._tokenize(word))
        return split_tokens


class LMAug:
    """
    基于mlm的数据增强，这里使用了wobert，对词mask
    """

    def __init__(self):
        model_path = 'E:\\ptm\\wobert'
        self.tokenizer = WoTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path).eval().to(get_device())

    def augmentation(self, text, topk=3):
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0]
        random_index = random.randint(1, len(input_ids) - 2)
        input_ids[random_index] = 103
        # mask_text = ''.join(segment)
        #
        # input_ids = self.tokenizer(mask_text, return_tensors='pt', max_length=512)['input_ids'].to(get_device())
        mask_index = [i for i, d in enumerate(input_ids) if d == 103]
        mask = input_ids == 103

        res = self.model(input_ids[None, :].to(get_device())).logits[0][mask]
        sort_res = torch.argsort(res, dim=1, descending=True)
        index = sort_res[:, 0:topk]

        out_text = []
        for idx in index.T:
            new_input_ids = input_ids
            for i, m_idx in zip(idx, mask_index):
                new_input_ids[m_idx] = i
            text = self.tokenizer.convert_ids_to_tokens(new_input_ids)
            text = ''.join(text[1:-1]).replace('#', '')
            out_text.append(text)
        return out_text


class Augmentation:
    """
    数据增强
    前4条是EDA，第5条是AEDA，第6条是回译，第7-9条是MLM，最后一条是GPT
    """

    def __init__(self, use_br=False, aug_list=None):
        if aug_list is None:
            aug_list = [
                EDA(),
                AEDA(),
                LMAug()
            ]
        if use_br:
            aug_list.append(BackTranslation())
        self.aug_list = aug_list

    def augmentation(self, text):
        aug_text = []
        for aug in self.aug_list:
            text_res = aug.augmentation(text)
            if isinstance(text_res, str):
                aug_text.append(text_res)
            elif isinstance(text_res, list):
                aug_text.extend(text_res)

        return aug_text


if __name__ == '__main__':
    # print(EDA().augmentation('今天天气真好啊'))
    # print(AEDA().augmentation('今天天气真好啊'))
    # print(BackTranslation().augmentation('今天天气真好啊'))
    print(LMAug().augmentation('今天天气真好啊', topk=5))
