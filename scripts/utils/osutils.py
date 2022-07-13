from __future__ import absolute_import

import os
import errno #标准的 errno 系统符号，每一个系统错误对应于一个整数

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def is_file(fname):
    return os.path.isfile(fname)

def is_dir(dirname):
    return os.path.isdir(dirname)

def join_path(path, *paths):
    return os.path.join(path, *paths)

class ObjectView(object):
    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        self.__dict__ = d