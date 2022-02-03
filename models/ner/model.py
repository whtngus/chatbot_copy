
# https://medium.com/@adam.wearne/intro-to-pytorch-with-nlp-b262e03bc8fa
from ner_data import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from tensorflow.keras import preprocessing
# https://github.com/TheAnig/NER-LSTM-CNN-Pytorch

torch.manual_seed(1)

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
    best_accuracy = 0
    for epoch in range(num_epochs):
        print("epoch : epoch")
        # Put model into training mode
        model.train()
        for sentence, tags in training_data:
            # Clear gradient
            model.zero_grad()
            # Prepare the sentence for network input
            input_sentence = prepare_sequence(sentence, word_to_ix)
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

        if cur_accuracy > best_accuracy:
            best_model = model

    torch.save(best_model.state_dict(), 'model.pt')

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
    ix_to_tag[10] = START_TAG
    ix_to_tag[0] = STOP_TAG
    tag_to_ix = {v: k for k, v in ix_to_tag.items()}


    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    num_epochs = 100
    model = AdamNet(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to GPU if we can
    model.to(device)
    do_train(model, num_epochs, train, test)

