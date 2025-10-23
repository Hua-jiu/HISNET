# -*-coding: utf-8 -*-
# -*-PowerByAnlong-*-
import os
import shutil



def delete_Dstore(path):  # Delete .DS_Store files
    """Delete .DS_Store files in macOS directories.

    Args:
        path (str): Directory path to clean DS_Store files.
    """
    for a, _, c in os.walk(path, topdown=True):
        for file in c:
            if file == '.DS_Store':
                os.remove(f'{a}/.DS_Store')


def file_copy(src, target, desame_flag=False):  # Copy file
    """
    Copy file from src to target.

    Args:
        src (str): source file path
        target (str): destination file path
        desame_flag (bool): if True, rename to avoid overwriting same-name file
    """
    if not os.path.exists(target):
        with open(src, 'rb') as rstream:
            contener = rstream.read()
            temp = text_segmentation(target, '/')
            temp.pop(-1)
            temp = text_segmentation(temp, '/', True)
            if not os.path.exists(temp):
                os.makedirs(temp)
                print('=' * 70)
                print(f'Create a new directory:{temp}')
                print('=' * 70)
            with open(target, 'wb') as wstream:
                wstream.write(contener)
    else:
        print('=' * 70)
        print(f'a same name file is aleady here:\nsrc:{src}\ntar:{target}')
        if desame_flag:
            temp = text_segmentation(target, '.')
            target = temp[0] + '_ds' + f'{temp[-1]}'
            with open(src, 'rb') as rstream:
                contener = rstream.read()
                with open(target, 'wb') as wstream:
                    wstream.write(contener)
            print('Now the new target file is: {}'.format(target))
            print('=' * 70)
        else:
            print('=' * 70)
        return False


def file_move(src, target):  # Move file
    """
    Move file from src to target.

    Args:
        src (str): source file path
        target (str): destination file path
    """
    file_copy(src, target, True)
    os.remove(src)


def text_segmentation(text, split, reverse=False):  # Text segmentation
    """
    Split or join text by a separator.

    Args:
        text (str | list): string or list
        split (str): separator
        reverse (bool): if True, join list into a string
    """
    temp = ''
    num = 0
    output = text
    n = text.count(split)  # Count occurrences of separator
    if reverse:
        for ele in text:
            if num == 0:
                temp += str(ele)
            else:
                temp += f'{split}{ele}'
            num += 1
        output = temp
        return output
    if n is None or n == 0:
        n = 0
        return [output]
    if n != 0:
        output = text.split(split, n)  # Perform splitting
    return output


def rename_dir_without_suffix(file_list, file_type, root_path):
    """
    Rename files in list without suffix indicator.
    """
    for file in file_list:
        if file.count('_') < 2:
            output = text_segmentation(file, split='.')
            new_filename = f'{root_path}/{output[0]}_s_{file_type}.{output[-1]}'  # maxilla _s_, mandible _m_
            try:
                os.rename(f'{root_path}/{file}', new_filename)
            except FileNotFoundError:
                continue  # May encounter duplicate files


def delete_blankdir(path):
    """
    Delete empty directories
    path: start directory
    """
    delete_Dstore(path)
    for root, dir_list, filename_list in os.walk(path, topdown=True):
        if len(filename_list) == 0 and len(dir_list) == 0:
            os.removedirs(root)


def count_file(path):
    """
    Count all files in directory
    """
    delete_Dstore(path)
    file_num = 0
    for root, dir_list, filename_list in os.walk(path, topdown=True):
        if len(filename_list):
            file_num += len(filename_list)
    return file_num


def rename_file(ori_file_path, new_filename):
    """
    Rename file
    ori_file_path: original file path
    new_filename: new filename
    """
    ori_filename = text_segmentation(ori_file_path, '/')[-1]
    temp = text_segmentation(ori_file_path, '/')
    temp[-1] = new_filename
    new_file_path = text_segmentation(temp, '/', reverse=True)
    os.rename(ori_file_path, new_file_path)
    print('=' * 70)
    print(f'{ori_filename} have been renamed to {new_filename}')
    print('=' * 70)