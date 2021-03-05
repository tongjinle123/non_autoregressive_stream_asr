import os
from os.path import join
from os.path import exists
import tarfile
import zipfile
import time
import pandas as pd
from tqdm import tqdm 



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