import pandas as pd
import torch
import time
import numpy as np
import warnings
import logging
import os

from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
#from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

"""
设置运行日志文件
通过调用 logger 类的实例来执行日志记录
"""
logging.basicConfig(
    filename='train-c.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger()


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    parser.add_argument('--epochs', default=60)
    parser.add_argument('--model', default='attention')
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = 'data/'
    lang = args.lang
    categories = 1
    if lang == 'java':
        categories = 5
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle(root+lang+'/train/blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle(root+lang+'/test/blocks.pkl').sample(frac=1)

    word2vec = Word2Vec.load(root+lang+"/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    # word2vec.syn0 是embedding得到的矩阵
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    #EPOCHS = 20
    EPOCHS = int(args.epochs)
    BATCH_SIZE = 32
    USE_GPU = True
    writer = SummaryWriter(
            f'runs/{EPOCHS}-epochs')
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings, model=args.model)
    logger.info(f'\t model: {args.model}, languge: {args.lang}, epoch: {EPOCHS}')
    
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    #print(train_data)
    precision, recall, f1 = 0, 0, 0
    print('Start training...')
    for t in range(1, categories+1):
        if lang == 'java':
            train_data_t = train_data[train_data['label'].isin([t, 0])]
            train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

            test_data_t = test_data[test_data['label'].isin([t, 0])]
            test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
        else:
            train_data_t, test_data_t = train_data, test_data
            #train_data_t, test_data_t = train_data[:200], test_data[:200]
        # training procedure
        print(f'EPOCHS:{EPOCHS}')
        start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            logger.info(f'Epoch #the {epoch+1} is starting! ')
            print(f'Epoch  {epoch+1} is starting!')
            #start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                #logger.info(f'\tLoss={loss.item()}')

                loss.backward()
                optimizer.step()
                writer.add_scalar(
                    'training_loss', loss.item(), epoch*len(train_data_t)+i+1)
            # 保存训练好的模型        
            model_save_dir = Path(f'saved_models/{args.model}')
            if not model_save_dir.exists():
                #model_save_dir.mkdir()
                os.makedirs(f'saved_models/{args.model}')
            torch.save(model.state_dict(), f'saved_models/{args.model}/model_epoch_'+str(epoch))
            #torch.save(model, 'saved_models/model_epoch_'+str(EPOCHS))
            print("Testing-%d..." % t)
        end_time = time.time()
        logger.info(f'train time: {end_time-start_time}')
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        models = os.listdir(f'saved_models/{args.model}')
        models.sort(key=lambda x: int(
                x.split('h')[1].split('_')[1]))
        print(models)
        writer_test = SummaryWriter(f'runs/{args.model}/{EPOCHS}_test')
        for index, save_model in enumerate(models):
            model.load_state_dict(torch.load(f'saved_models/{args.model}/{save_model}'))
            model.eval()
            while i < len(test_data_t):
                batch = get_batch(test_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                test1_inputs, test2_inputs, test_labels = batch
                if USE_GPU:
                    test_labels = test_labels.cuda()

                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                output = model(test1_inputs, test2_inputs)

                loss = loss_function(output, Variable(test_labels))
                #writer_test.add_scalar('loss', loss, index+1)

                # calc testing acc
                predicted = (output.data > 0.5).cpu().numpy()
                predicts.extend(predicted)
                trues.extend(test_labels.cpu().numpy())
                total += len(test_labels)
                total_loss += loss.item() * len(test_labels)
            if lang == 'java':
                weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
                p, r, f, _ = precision_recall_fscore_support(
                    trues, predicts, average='binary')
                precision += weights[t] * p
                recall += weights[t] * r
                f1 += weights[t] * f
                print("Type-" + str(t) + ": " + str(p) +
                    " " + str(r) + " " + str(f))
                logger.info(f'\tP: {p}, R: {r}, F1: {f}')
                logger.info(f'\tP: {precision}, R: {recall}, F1: {f1}')
            else:
                # precision, recall, f1, support = precision_recall_fscore_support(
                #     trues, predicts, average='binary')
                cm = confusion_matrix(trues, predicts)
                acc = accuracy_score(trues, predicts)
                precision = precision_score(
                    trues, predicts, average="weighted")
                recall = recall_score(trues, predicts, average="weighted")
                f1 = f1_score(trues, predicts, average="weighted")
                #score = roc_auc_score(y_true, y_pred)
                report = classification_report(trues, predicts)
                logger.info(f'\tP: {precision}, R: {recall}, F1: {f1}, ACC: {acc}')
            
            #writer_test.add_scalar('test_acc', acc, index)
            #acc = accuracy_score(trues, predicts)
            writer_test.add_scalar('test_accuracy', acc, index+1)
            writer_test.add_scalar('test_precision', precision, index+1)
            writer_test.add_scalar('test_recall', recall, index+1)
            writer_test.add_scalar('test_f1', f1, index+1)
            
            print(f'\tP: {precision}, R: {recall}, F1: {f1}, acc: {acc}')
        
    # print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" %
    #       (precision, recall, f1))
    # print(f'acc: {acc}')
            # logger.info(
            #     f'\t testing results(P,R,F1,acc) for {save_model}: \np:{precision}, R:{recall}, F1:{f1}, ACC:{acc}')
