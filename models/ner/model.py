
# https://medium.com/@adam.wearne/intro-to-pytorch-with-nlp-b262e03bc8fa
# from ner_data import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from tensorflow.keras import preprocessing
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.manual_seed(1)
import sys, os
from utils.Preprocess import Preprocess
from sklearn.model_selection import train_test_split

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

class AdamNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):

        # Required call to the constructor of the parent class
        super(AdamNet, self).__init__()

        # Dimension of word embeddings, and the LSTM's hidden state vector
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Embedding layer to turn our vocab into dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Fully-connected layer that we'll use for prediction
        self.fc = nn.Linear(hidden_dim, tagset_size)


    def forward(self, sentence):
        embeddings = self.embeddings(sentence)

        lstm_output, hidden = self.lstm(embeddings.view(len(sentence), 1, -1))

        raw_scores = self.fc(lstm_output.view(len(sentence), -1))

        tag_scores = F.log_softmax(raw_scores, dim=1)

        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def do_train(model, num_epochs,train, dev):
    # Training Loop
    best_loss = 999999999999999
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        print("epoch : {}".format(epoch))
        # Put model into training mode
        model.train()
        for i, (sentence, tags) in enumerate(tqdm(training_data)):
            # Clear gradient
            try:
                # if i == 41219 or i == 37721:
                #     continue
                # 41219
                model.zero_grad()
                # Prepare the sentence for network input
                input_sentence = torch.tensor(sentence, dtype=torch.long)
                targets = prepare_sequence(tags, tag_to_ix)
                # Move the data over to the GPU
                input_sentence = input_sentence.to(device)
                targets = targets.to(device)
                # Run the forward pass
                tag_scores = model(input_sentence)
                # Calculate the loss
                loss = loss_function(tag_scores, targets)
                # Backward pass
                loss.backward()
                # Update model parameters
                optimizer.step()
                loss = loss.item()
                epoch_loss += loss
                writer.add_scalar("Loss/train_step", loss, step)
                step+=1
                if i % 100 == 0 and i != 0:
                    print(f"loss = {epoch_loss/i}")
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e)
                print()
        torch.save(model.state_dict(), f'./model/model_{epoch}_{epoch_loss}.pt')
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            best_model = model
        writer.add_scalar("Loss/train", epoch_loss, epoch)
    torch.save(best_model.state_dict(), './model/best_model.pt')

def do_eval():
    N = 901
    model.eval()

    inputs = prepare_sequence(dataset[N][0], word_to_ix)
    inputs = inputs.to(device)

    tag_scores = model(inputs)

    ix_to_tag = dict((v, k) for k, v in tag_to_ix.items())

    preds = [torch.max(x, 0)[1].item() for x in tag_scores]
    correct = prepare_sequence(dataset[N][1], tag_to_ix)

    original_sentence = dataset[N][0]
    correct_tags = [ix_to_tag[c.item()] for c in correct]
    predicted_tags = [ix_to_tag[p] for p in preds]

    print('{:<15}|{:<15}|{:<15}\n'.format(*['Original', 'Correct', 'Predicted']))
    correct_tags_count = 0
    for item in zip(original_sentence, correct_tags, predicted_tags):
        print('{:<15}|{:<15}|{:<15}'.format(*item))
        correct_tags_count += correct_tags
    return correct_tags_count


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

if __name__ == "__main__":
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 30
    p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
                   userdic='../../utils/user_dic.tsv')

    # 학습용 말뭉치 데이터를 불러옴
    corpus = read_file('ner_train.txt')
    sentences, tags = get_corpus('ner_train.txt')

    # 토크나이저 정의
    tag_tokenizer = preprocessing.text.Tokenizer(lower=False)  # 태그 정보는 lower=False 소문자로 변환하지 않는다.
    tag_tokenizer.fit_on_texts(tags)

    training_data = [(p.get_wordidx_sequence(s), t) for s, t in zip(sentences, tags)]

    word_to_ix = p.word_index

    train, test= train_test_split(training_data, test_size=.1, random_state=1234)

    ix_to_tag = tag_tokenizer.index_word
    ix_to_tag = {k-1:v for  k,v in ix_to_tag.items()}
    tag_to_ix = {v: k for k, v in ix_to_tag.items()}


    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    num_epochs = 10
    model = AdamNet(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to GPU if we can
    model.to(device)
    do_train(model, num_epochs, train, test)
    writer.flush()
    writer.close()


