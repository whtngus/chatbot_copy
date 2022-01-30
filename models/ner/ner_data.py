import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from konlpy.tag import Komoran

class Preprocess:
    def __init__(self, word2index_dic='', userdic=None):
        # 단어 인덱스 사전 불러오기
        if(word2index_dic != ''):
            f = open(word2index_dic, "rb")
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

        # 형태소 분석기 초기화
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사
        # 참조 : https://docs.komoran.kr/firststep/postypes.html
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    # 형태소 분석기 POS 태거
    def pos(self, sentence):
        jpype.attachThreadToJVM()
        return self.komoran.pos(sentence)

    # 불용어 제거 후, 필요한 품사 정보만 가져오기
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list

    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []

        w2i = []
        for word in keywords:
            if word in self.word_index:
                w2i.append(word)
            else:
                # 해당 단어가 사전에 없는 경우, OOV 처리
                w2i.append('OOV')
        return w2i


# 학습 파일 불러오기
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents




p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')

# 학습용 말뭉치 데이터를 불러옴

def get_corpus(path):
    corpus = read_file(path)
    # 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
    sentences, tags = [], []
    for t in corpus:
        tagged_sentence = []
        sentence, bio_tag = [], []
        for w in t:
            tagged_sentence.append((w[1], w[3]))
            sentence.append(w[1])
            bio_tag.append(w[3])

        sentences.append(sentence)
        tags.append(bio_tag)
    return sentences, tags



# 학습용 단어 시퀀스 생성
# x_train = [p.get_wordidx_sequence(sent) for sent in sentences]
# y_train = tag_tokenizer.texts_to_sequences(tags)

# index_to_ner =  # 시퀀스 인덱스를 NER로 변환 하기 위해 사용
# index_to_ner[0] = 'PAD'

# 시퀀스 패딩 처리
# max_len = 40
# x_train = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
# y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)

# 학습 데이터와 테스트 데이터를 8:2의 비율로 분리

# # 출력 데이터를 one-hot encoding
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)
#
# print("학습 샘플 시퀀스 형상 : ", x_train.shape)
# print("학습 샘플 레이블 형상 : ", y_train.shape)
# print("테스트 샘플 시퀀스 형상 : ", x_test.shape)
# print("테스트 샘플 레이블 형상 : ", y_test.shape)
