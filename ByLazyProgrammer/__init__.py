# -*- coding: utf-8 -*-
# @Time    : @Date
# @Author  : CJ
import logging
import pandas as pd
import os
import numpy as np

os.environ['TZ'] = 'GMT'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s | %(message)s"
logging.basicConfig(filename='../temp/my.log', level=logging.INFO, format=LOG_FORMAT)

if __name__ == '__main__':
    pass