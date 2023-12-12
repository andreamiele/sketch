import numpy as np
import os
import sys
import csv
import numpy as np
import json
import pickle  # python3.x
import time
from datetime import timedelta, datetime
import subprocess
import struct
import errno
from pprint import pprint
import glob

def lines_to_strokes(lines, omit_first_point=True):
    """
    Convert polyline format to stroke-3 format.
    lines: list of strokes, each stroke has format Nx2
    """
    strokes = []
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :] if omit_first_point else strokes

def strokes_to_lines(strokes, scale=1.0, start_from_origin=False):
    """
    convert strokes3 to polyline format ie. absolute x-y coordinates
    note: the sketch can be negative
    :param strokes: stroke3, Nx3
    :param scale: scale factor applied on stroke3
    :param start_from_origin: sketch starts from [0,0] if True
    :return: list of strokes, each stroke has format Nx2
    """
    x = 0
    y = 0
    lines = []
    line = [[0, 0]] if start_from_origin else []
    for i in range(len(strokes)):
        x_, y_ = strokes[i, :2] * scale
        x += x_
        y += y_
        line.append([x, y])
        if strokes[i, 2] == 1:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
            line = []
    if lines == []:
        line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
        lines.append(line_array)
    return lines

def load_pickle(path):
    """
    load a pickled object
    :param path: .pkl path
    :return: the pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class Tokenizer(object):
    """
    tokenize sketches in stroke3 using clustering
    """

    def __init__(self, dict_path, max_seq_len=0):
        """
        initialize dictionary (a sklearn cluster object)
        :param dict_path: path to pickle file
        :param max_seq_len: 0 if variable length sketch
        """
        self.max_seq_len = max_seq_len
        self.dict = load_pickle(dict_path)
        self.PAD = 0
        self.SEP = self.dict.n_clusters + 1  # whole dictionary needs to be shifted by 1
        self.SOS = self.dict.n_clusters + 2
        self.EOS = self.dict.n_clusters + 3
        self.VOCAB_SIZE = self.dict.n_clusters + 4

    def encode(self, stroke3, seq_len=0):
        """
        encode stroke3 into tokens
        :param stroke3: stroke3 data as numpy array (nx3)
        :param seq_len: if positive, the output is padded with PAD
        :return: sequence of integers as list
        """
        stroke3 += np.zeros((1, 3))
        out = self.dict.predict(stroke3[:, :2])
        # shift by 1 due to PAD token
        out = out + 1
        out = list(out)
        # insert SEP token
        positions = np.where(stroke3[:, 2] == 1)[0]
        offset = 1
        for i in positions:
            out.insert(i + offset, self.SEP)
            offset += 1
        # insert SOS and EOS
        out = [self.SOS] + out + [self.EOS]
        if self.max_seq_len:  # pad
            npad = self.max_seq_len - len(out)
            if npad > 0:
                out += [self.PAD] * npad
            else:
                out = out[:self.max_seq_len]
                out[-2:] = [self.SEP, self.EOS]
        if len(out) < seq_len:
            out += [self.PAD] * (seq_len-len(out))
        return np.array(out)

    def decode(self, seqs):
        if len(seqs) > 0 and isinstance(seqs[0], (list, tuple, np.ndarray)):
            return self.decode_list(seqs)
        else:
            return self.decode_single(seqs)

    def decode_single(self, seq):
        """
        decode a sequence of token id to stroke3
        :param seq: list of integer
        :return: stroke3 array (nx3)
        """
        cluster_ids = []
        pen_states = []
        for i in seq:
            if i not in [self.SOS, self.EOS, self.SEP, self.PAD]:
                cluster_ids.append(i)
                pen_states.append(0)
            elif i == self.SEP and len(pen_states) > 0:
                pen_states[-1] = 1
            elif i == self.EOS:
                break
        if len(cluster_ids) > 0:
            cluster_ids = np.array(cluster_ids)
            cluster_ids = cluster_ids - 1
            dxy = self.dict.cluster_centers_[cluster_ids]
            out = np.c_[dxy, np.array(pen_states)]
            return np.array(out)
        else:
            return np.zeros((1, 3), dtype=np.float32)  #empty sketch

    def decode_list(self, sketches):
        decoded = []
        for s in sketches:
            decoded.append(self.decode_single(np.squeeze(s)))
        return decoded