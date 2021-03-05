from torch import HalfStorage
from src.tools.data_module import Builder
import os 
from src.model.module import Featurizer
from src.tools.vocab import Vocab
import pandas as pd 
from src.tools.tokenizer import combine, tokenize
import torchaudio as ta
import numpy as np 
from tqdm import tqdm 

def line_iter(manifest_file, featuizer, vocab, rate=1.0):
    df = pd.read_csv(manifest_file)
    df = df[['wav_file', 'target']]
    # df = df.head(1000)
    for index in tqdm(range(len(df)), desc=f'{manifest_file}'):
        data = df.loc[index]
        feature, feature_length = featurizer.line(data['wav_file'], rate=rate)
        feature = feature[0]
        feature_length, feature_size = feature.shape
        target = np.array(vocab.str2id(combine(tokenize(data['target']))), dtype=np.int32)
        if feature_length < 1600 and len(target) < 150:
            yield (feature.reshape(-1).numpy(), feature_length, feature_size, target)
        else:
            print(f'bad line: {manifest_file}, {index}')
            yield None

if __name__ == '__main__':
    vocab = Vocab('vocab.model')
    featurizer = Featurizer()
    featurizer.load_mean_std('mean_std.pth')
    csv_root = 'data/manifest'
    hdf_root = 'data/hdf'
    FILES = [
        'data_aishell_train.csv','data_aishell_dev.csv','data_aishell_test.csv', 
        'aidatatang_200zh.csv',
        'AISHELL-2.csv','c_500.csv','magic_data_train.csv','magic_data_dev.csv',
        'magic_data_test.csv','prime.csv','stcmds.csv',
        'ce_200.csv',
        'ce_20_dev.csv','libri_100.csv','libri_360.csv','libri_500.csv'
        ]
    hdf_files = [os.path.join(hdf_root, i.replace('.csv', '.hdf')) for i in FILES]
    files = [os.path.join(csv_root, i) for i in FILES]
    builder = Builder()
    for file, hdf_file in zip(files, hdf_files):
        builder.build(hdf_file, line_iter(file, featurizer, vocab))

    # FILES = [
    #     # 'data_aishell_train.csv','data_aishell_dev.csv','data_aishell_test.csv', 
    #     # 'aidatatang_200zh.csv',
    #     # 'AISHELL-2.csv','c_500.csv','magic_data_train.csv','magic_data_dev.csv',
    #     # 'magic_data_test.csv','prime.csv','stcmds.csv',
    #     'ce_200.csv',
    #     # 'ce_20_dev.csv','libri_100.csv','libri_360.csv','libri_500.csv'
    #     ]
    # hdf_files = [os.path.join(hdf_root, i.replace('.csv', '.hdf')) for i in FILES]
    # files = [os.path.join(csv_root, i) for i in FILES]
    
    
    # file = 'ce_200.csv'
    # builder = Builder()
    # for rate in [0.9, 0.95, 1.05, 1.1]:
    #     hdf_file = os.path.join(hdf_root, str(int(rate*100)) + '_' + file.replace('.csv','.hdf'))
    #     builder.build(hdf_file, line_iter(os.path.join(csv_root, file), featurizer, vocab, rate))

    # for file, hdf_file in zip(files, hdf_files):
    #     builder.build(hdf_file, line_iter(file, featurizer, vocab))



    
