import os
from os.path import join
from os.path import exists
import tarfile
import zipfile
import pandas as pd
from tqdm import tqdm 
import json 
from config import DATA_ROOT, RAW_FOLDER, EXTRACTED_FOLDER, MANIFEST_FOLDER, CKPT_ROOT,HDF_FOLDER


def my_mkdir(path: str):
    if not exists(path):
        os.mkdir(path)
        print(f'{path} created')
    else:
        print(f'{path} already exists')

def init_experiments():
    my_mkdir(DATA_ROOT)
    my_mkdir(RAW_FOLDER)
    my_mkdir(EXTRACTED_FOLDER)
    my_mkdir(MANIFEST_FOLDER)
    my_mkdir(CKPT_ROOT)
    my_mkdir(HDF_FOLDER)

# def preprocess_single_wav(file):
#     data = preprocess_wav(file)
#     write_wav(file, data)
#     return data.shape[0] / 16000

def process_raw_datas():
    pass

def process_aishell():

    def extract_target(file):
        with open(file, encoding='utf8') as reader:
            data = reader.readlines()
            name2target = {i.strip().split(' ')[0]:''.join(i.strip().split(' ')[1:]) for i in data}
        return name2target

    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]

    raw_file = join(RAW_FOLDER, 'data_aishell.tar')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list[0])
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_libri():
    def extract_target(file_list):
        name2target = {}
        for file in tqdm(file_list, desc='filelist'):
            with open(file, encoding='utf8') as reader:
                data = reader.readlines()
                for line in data:
                    name = line.split(' ')[0]
                    target = ' '.join(line.strip().split(' ')[1:]).lower()
                    name2target[name] = target
        return name2target

    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    for raw_file in [join(RAW_FOLDER, i) for i in ['libri_500.tar.gz','libri_360.tar.gz','libri_100.tar.gz']]:
        prefix = raw_file.split('/')[-1].split('.')[0]
        type = 'zip' if raw_file.endswith('.zip') else 'tar'
        folder_extract_to = join(EXTRACTED_FOLDER, prefix)
        manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
        extract_nested_file(raw_file, folder_extract_to, type)
        wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.flac')
        txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
        target_dict = extract_target(txt_list)
        merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
        print('done')

def process_c_500():
    def extract_target(file_list):
        name2target = {}
        for file in tqdm(file_list, desc='filelist'):
            name = file.split('/')[-1].split('.')[0]
            with open(file, encoding='utf8') as reader:
                data = reader.readline()
            target = data.strip()
            name2target[name] = target
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    raw_file = join(RAW_FOLDER, 'c_500.zip')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list)
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_ce200():
    def extract_target(file_list):
        name2target = {}
        for file in file_list:
            name = file.split('/')[-1].split('.')[0]
            with open(file, encoding='utf8') as reader:
                data = reader.readline()
            target = data.strip()
            name2target[name] = target
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    raw_file = join(RAW_FOLDER, 'ce_200.tar.gz')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list)
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_datatang():
    def extract_target(file_list):
        name2target = {}
        for file in tqdm(file_list):
            name = file.split('/')[-1].split('.')[0]
            with open(file, encoding='utf8') as reader:
                data = reader.readline()
            target = data.strip()
            name2target[name] = target
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    raw_file = join(RAW_FOLDER, 'aidatatang_200zh.tgz')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list)
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_magic():
    
    def extract_target(file):
        with open(file, encoding='utf8') as reader:
            data = reader.readlines()[1:]
            name2target = {i.strip().split('\t')[0].split('.')[0]: ''.join(i.strip().split('\t')[-1]) for i in data}
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    for raw_file in ['magic_data_train.tar.gz', 'magic_data_test.tar.gz', 'magic_data_dev.tar.gz']:
        raw_file = os.path.join(RAW_FOLDER, raw_file)
        prefix = raw_file.split('/')[-1].split('.')[0]
        type = 'zip' if raw_file.endswith('.zip') else 'tar'
        folder_extract_to = join(EXTRACTED_FOLDER, prefix)
        manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
        extract_nested_file(raw_file, folder_extract_to, type)
        wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
        txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
        target_dict = extract_target(txt_list[0])
        merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
        print('done')

def process_aishell2():
    def extract_target(file):
        with open(file, encoding='utf8') as reader:
            data = reader.readlines()
            name2target = {i.strip().split('\t')[0]:i.strip().split('\t')[-1] for i in data}
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    raw_file = join(RAW_FOLDER, 'AISHELL-2.zip')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list[0])
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_prime():
    def extract_target(file):
        name2target = {}
        data = json.load(open(txt_list[0], encoding='utf8'))
        for line in data:
            name2target[line['file'].split('.')[0]] = ''.join(line['text'].split(' '))
        return name2target
    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]

    raw_file = join(RAW_FOLDER, 'prime.tar.gz')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.json')
    target_dict = extract_target(txt_list[0])
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')

def process_stcmd():

    def extract_target(file_list):
        name2target = {}
        for file in file_list:
            name = file.split('/')[-1].split('.')[0]
            with open(file, encoding='utf8') as reader:
                data = reader.readline()
            target = data.strip()
            name2target[name] = target
        return name2target

    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    
    raw_file = join(RAW_FOLDER, 'stcmds.tar.gz')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    target_dict = extract_target(txt_list)
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')


def process_ce20_dev():
    def extract_target(file_list):
        name2target = {}
        for file in file_list:
            name = file.split('/')[-1].split('.')[0]
            with open(file, encoding='utf8') as reader:
                data = reader.readline()
            target = data.strip()
            name2target[name] = target
        return name2target


    def extract_name_fn(path):
        return path.split('/')[-1].split('.')[0]
    
    raw_file = join(RAW_FOLDER, 'ce_20_dev.zip')
    prefix = raw_file.split('/')[-1].split('.')[0]
    type = 'zip' if raw_file.endswith('.zip') else 'tar'
    folder_extract_to = join(EXTRACTED_FOLDER, prefix)
    manifest_csv_path = join(MANIFEST_FOLDER, prefix + '.csv')
    extract_nested_file(raw_file, folder_extract_to, type)
    wav_list = search_folder_for_post_fix_file_list(folder_extract_to, '.wav')
    txt_list = search_folder_for_post_fix_file_list(folder_extract_to, '.txt')
    txt_list = [i for i in txt_list if '__MACOSX' not in i]
    target_dict = extract_target(txt_list)
    print('here')
    merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)
    print('done')






#________________________________


def search_folder_for_post_fix_file_list(root, post_fix):
    """
    search folder recurrnetly for files with post_fix
    :param root: path
    :param post_fix: post_fix
    :return: file list
    """
    targets = []
    files = [join(root, i) for i in os.listdir(root)]
    for i in files:
        if i.endswith(post_fix):
            targets.append(i)
        if os.path.isdir(i):
            for j in search_folder_for_post_fix_file_list(i, post_fix):
                targets.append(j)
    return targets

def extract_file(file_to_extract, folder_extract_to, type):
    """
    extract zip of tar.gz file
    :param file_to_extract:
    :param folder_extract_to:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    tools = {'zip': zipfile.ZipFile, 'tar': tarfile}
    if type == 'tar':
        with tools[type].open(file_to_extract) as file:
            file.extractall(folder_extract_to)
    else:
        with tools[type](file_to_extract) as file:
            file.extractall(folder_extract_to)
    print('extract_file done')

def nested_extract(extracted_folder, type):
    """
    extract zip or tar.gz files which deep in a folder
    :param extracted_folder:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    current_folder = extracted_folder
    current_files = [join(current_folder, i) for i in os.listdir(current_folder)]

    for file in current_files:
        if os.path.isdir(file):
            nested_extract(file, type)
        if file.endswith('.tar.gz'):
            extract_file(file, current_folder, type)
            os.remove(file)

def extract_nested_file(file_to_extract, folder_extract_to, type):
    """
    extract file that have zipped file in zip file
    :param file_to_extract:
    :param folder_extract_to:
    :param type:
    :return:
    """
    assert type in ['zip', 'tar']
    extract_file(file_to_extract, folder_extract_to, type)
    nested_extract(folder_extract_to, type)
    print('extract_nested_file done ')

def merge(wav_list, target_dict, extract_name_fn, manifest_csv_path):
    wav_df = pd.DataFrame(wav_list, columns=['wav_file'])
    wav_df.index = wav_df.wav_file.apply(extract_name_fn)
    target_df = pd.DataFrame.from_dict(target_dict, orient='index', columns=['target'])
    merged_df = pd.merge(left=wav_df, right=target_df, left_index=True, right_index=True)
    try:
        merged_df.to_csv(manifest_csv_path, encoding='utf8')
    except:
        merged_df.to_csv(manifest_csv_path)
    print(f'manifest saved to {manifest_csv_path}')
    return 'done'

def extract_name_fn(path):
    return path.split('/')[-1].split('.')[0]

# def cal_duration(df_file):
#     df = pd.read_csv(df_file)
#     df['duration'] = df.wav_file.apply()

if __name__ == '__main__':
    init_experiments()
    process_aishell()
    process_libri()
    process_c_500()
    process_ce200()
    process_datatang()
    process_magic()
    process_ce20_dev()
    process_aishell2()
    process_stcmd()
    process_prime()

    ## split data_aishell
    import pandas as pd 
    import os 
    df = pd.read_csv('data/manifest/data_aishell.csv')
    for i in ['train','test','dev']:
        df[df.wav_file.apply(lambda x: i in x)].to_csv(f'data/manifest/data_aishell_{i}.csv')
    os.remove('data/manifest/data_aishell.csv')


    


    