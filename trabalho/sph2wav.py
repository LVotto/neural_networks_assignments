# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:47:13 2018

@author: luiz_
"""

from sphfile import SPHFile
from pathlib import Path
import os

directory_in_str = './trabalho/TIMIT'

pathlist = Path(directory_in_str).glob('**/*.wav')
for path in pathlist:
    f = open(str(path), 'rb')
    head = next(f)
    f.close()
    # If format is NIST SPHERE, convert to correct WAV
    if b'NIST' in head:
        path_str = str(path)
        sph = SPHFile(path_str)
        sph.write_wav(path_str[:-4] + 'w.wav')
        os.rename(path, str(path)[:-4] + 's.sph')