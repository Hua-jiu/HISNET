# -*-coding: utf-8 -*-
# -*-PowerByAnlong-*-
import os
import shutil



def delete_Dstore(path):  # 删除.Ds_Store文件
    """to delete the Dstore file in MacOS

    Args:
        path (str): the path of the dir you want to delete DS_Store
    """
    for a, _, c in os.walk(path, topdown=True):
        for file in c:
            if file == '.DS_Store':
                os.remove(f'{a}/.DS_Store')


def file_copy(src, target, desame_flag=False):  # 复制文件
    """
    src:原文件地址
    target:目标文件地址
    desame_flag:是否重命名文件已保证不替换原名文件
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


def file_move(src, target):  # 剪切文件
    """
    src:原文件地址
    target:目标文件地址
    """
    file_copy(src, target, True)
    os.remove(src)


def text_segmentation(text, split, reverse=False):  # 进行文本分割
    """
    根据特定符号进行文本分割/组合
    text:字符串/数组
    split:指定符号
    reverse:为TRUE时将数组重新组合成字符串
    """
    temp = ''
    num = 0
    output = text
    n = text.count(split)  # 统计文本分割次数
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
        output = text.split(split, n)  # 进行文本分割
    return output


def rename_dir_without_suffix(file_list, file_type, root_path):
    """
    重命名文件夹
    """
    for file in file_list:
        if file.count('_') < 2:
            output = text_segmentation(file, split='.')
            new_filename = f'{root_path}/{output[0]}_s_{file_type}.{output[-1]}'  # 上颌骨_s_ 下颌骨_m_
            try:
                os.rename(f'{root_path}/{file}', new_filename)
            except FileNotFoundError:
                continue  # 可能遍历到重复文件


def delete_blankdir(path):
    """
    删除空白文件夹
    path:开始遍历的起始地址
    """
    delete_Dstore(path)
    for root, dir_list, filename_list in os.walk(path, topdown=True):
        if len(filename_list) == 0 and len(dir_list) == 0:
            os.removedirs(root)


def count_file(path):
    """
    统计文件夹中所有文件的个数
    """
    delete_Dstore(path)
    file_num = 0
    for root, dir_list, filename_list in os.walk(path, topdown=True):
        if len(filename_list):
            file_num += len(filename_list)
    return file_num


def rename_file(ori_file_path, new_filename):
    """
    重命名文件
    ori_file_path:原始文件地址
    new_filename:文件新名称
    """
    ori_filename = text_segmentation(ori_file_path, '/')[-1]
    temp = text_segmentation(ori_file_path, '/')
    temp[-1] = new_filename
    new_file_path = text_segmentation(temp, '/', reverse=True)
    os.rename(ori_file_path, new_file_path)
    print('=' * 70)
    print(f'{ori_filename} have been renamed to {new_filename}')
    print('=' * 70)