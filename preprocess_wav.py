import pandas as pd 
from src.tools.vad import preprocess_wav, write_wav
import os 
from tqdm import tqdm 
from src.model.module import Featurizer
import torch as t


MANIFESTROOT = 'data/manifest'
MANIFEST = [os.path.join(MANIFESTROOT, i) for i in os.listdir(MANIFESTROOT) if i.endswith('.csv')]
MANIFEST = list(filter(lambda x: 'libri' not in x, MANIFEST))

# def vad_and_write(wav_file):
#     sig = preprocess_wav(wav_file, source_sr=16000)
#     file = write_wav(wav_file, sig)
#     return file


if __name__ == '__main__':
    # for file in tqdm(MANIFEST, desc='file'):
    #     df = pd.read_csv(file)
    #     for wav in tqdm(df.wav_file, desc=f'{file}:wav'):
    #         vad_and_write(wav)

    featurizer = Featurizer()
    df = pd.DataFrame()
    for file in MANIFEST:
        ndf = pd.read_csv(file)
        df = pd.concat([df, ndf])
    
    samples = df.sample(50000)
    wav_files = samples.wav_file
    m, s = featurizer.cal_cmvn(wav_files)
    t.save({'mean': m, 'std': s}, 'mean_std.pth')

    

