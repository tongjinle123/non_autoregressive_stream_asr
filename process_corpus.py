
import pandas as pd 
from src.tools.tokenizer import tokenize, is_chinese, combine
import os 
from os.path import join 
from start_experiment import MANIFEST_FOLDER
from tqdm import tqdm, tqdm_pandas
from collections import Counter
import sentencepiece as spm
from start_experiment import MANIFEST_FOLDER
from tqdm import tqdm 
tqdm_pandas(tqdm)
CORPUS = 'data/corpus.txt'
ENG_CORPUS = 'data/eng_corpus.txt'

TRAIN_MANIFEST_LIST_FORCORPUS_CH = [
    'aidatatang_200zh.csv' ,'AISHELL-2.csv', 'c_500.csv', 'ce_200.csv','data_aishell.csv', 'magic_data_train.csv', 
    'magic_data_dev.csv','magic_data_test.csv','stcmds.csv','prime.csv']
TRAIN_MANIFEST_LIST_FORCORPUS_CH = [join(MANIFEST_FOLDER, i) for i in TRAIN_MANIFEST_LIST_FORCORPUS_CH]

TRAIN_MANIFEST_LIST_FORCORPUS_EN = ['libri_100.csv','libri_360.csv', 'libri_500.csv']
TRAIN_MANIFEST_LIST_FORCORPUS_EN = [join(MANIFEST_FOLDER, i) for i in TRAIN_MANIFEST_LIST_FORCORPUS_EN]

DEV_MANIFEST_LIST_FORCORPUS = [join(MANIFEST_FOLDER, 'ce_20_dev.csv')]


def collect_corpus(extern_ch_corpus=None, extern_en_corpus=None):
    with open(CORPUS,'w') as writer: 
        for file in tqdm(TRAIN_MANIFEST_LIST_FORCORPUS_CH):
            df = pd.read_csv(file)
            for line in df.target:
                line = combine(tokenize(line))
                writer.write(line.strip() + '\n')
        if extern_ch_corpus:
            with open(extern_ch_corpus) as reader:
                for line in reader.readlines():
                    line = combine(tokenize(line))
                    writer.write(line.strip() + '\n')

    with open(ENG_CORPUS, 'w') as writer:
        for file in tqdm(TRAIN_MANIFEST_LIST_FORCORPUS_EN):
            df = pd.read_csv(file)
            for line in df.target:
                writer.write(line.strip() + '\n')
        if extern_en_corpus:
            with open(extern_en_corpus) as reader:
                for line in reader.readlines():
                    writer.write(line.strip() + '\n')

def build_vocab():
    chinese_word_list = []
    chinese_counter = Counter()
    with open(CORPUS) as reader:
        for i in reader.readlines():
            for j in i:
                if is_chinese(j):
                    chinese_counter.update(j)
    
    chinese_chars = ''
    for i in chinese_counter.most_common(4200):
        chinese_chars += i[0].strip() + ','
    chinese_chars = chinese_chars[:-1]
    config_string = '--split_by_whitespace=1 --normalization_rule_name=nmt_nfkc_cf \
        --add_dummy_prefix=1 --model_type=bpe --model_prefix=vocab --input=data/eng_corpus.txt --vocab_size=6000 \
            --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
                --user_defined_symbols=[S],[Bs],[N],[T],[P],[PCH],[PEN],' + chinese_chars[:-1] + "\'s,\'re,\'ve,\'ll,\'t,\'d,\'m"  #,\'re,\'ve,\'ll,\'t,\'d,\'m,o\',\'ll"
    print(config_string)
    vocab_trainer = spm.SentencePieceTrainer.Train(config_string)
    print(config_string)
    return chinese_counter


if __name__ == '__main__':
    collect_corpus()
    build_vocab()