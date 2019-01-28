#import sys
#sys.path.append('C:/Users/PelaoT/Desktop/Practica/codigo')

import io
import os
import tarfile

import cv2
import numpy as np

from common import utils


def join_path(folder, file):
    last_char_pos = len(folder)
    last_char = folder[last_char_pos - 1]
    if last_char == '/':
        return folder + file
    else:
        return folder + '/' + file


class ImCutter:
    def __init__(self, save_dir='', cut_size=63,
                 save_as_tar=False):

        if save_dir[len(save_dir) - 1] != '/':
            save_dir += '/'

        self._saveDir = save_dir
        self._save_count = 1
        self._cut_size = int(cut_size / 2)
        self.save_as_tar = save_as_tar
        tar_name = save_dir + "candidates.tar"
        self.purge_previous_im(tar_name)
        if save_as_tar:
            self.tar_name = tar_name
            self.tar = tarfile.TarFile(name=self.tar_name, mode="a")

    def cut_and_save(self, col_center, row_center, save_to_disk=True):
        left, right, top, bottom = self.get_borders(row_center, col_center)
        # numpy indexing is [inclusive:exclusive], that's why i add 1
        cut = self._im[top:bottom + 1, left:right + 1]
        cut2 = self.pad_image(cut, row_center, col_center)

        if save_to_disk:
            save_name = self._base_name + '-' + str(self._save_count) + '.png'
            if self.save_as_tar:
                self.save_to_tar(save_name, cut2)
                pass
            else:
                # i don't use os.path.join because opencv uses unix style path even on windows
                save_path = join_path(self._saveDir, save_name)
                cv2.imwrite(save_path, cut2)
            self._save_count += 1

    def get_borders(self, row_center, col_center):
        shape = self._im.shape
        cols = shape[0]
        rows = shape[1]

        # checks that borders are inside image
        left = max(col_center - self._cut_size, 0)
        right = min(col_center + self._cut_size, cols)
        top = max(row_center - self._cut_size, 0)
        bottom = min(row_center + self._cut_size, rows)

        # for some reason doesnt slice well on top borders
        if right == cols:
            left -= 1

        if bottom == rows:
            top -= 1

        return left, right, top, bottom

    def pad_image(self, cutted, row_center, col_center):
        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0
        cols = self._im.shape[0]
        rows = self._im.shape[1]

        shape = cutted.shape
        if shape[0] == self._cut_size * 2 + 1 and shape[1] == self._cut_size * 2 + 1:
            return cutted

        # top boundary
        if row_center - self._cut_size < 0:
            top_pad = self._cut_size - row_center

        # bottom boundary
        if row_center + self._cut_size > rows:
            bottom_pad = self._cut_size - (rows - row_center)

        # left boundary
        if col_center - self._cut_size < 0:
            left_pad = self._cut_size - col_center

        # right boundary
        if col_center + self._cut_size > cols:
            right_pad = self._cut_size - (cols - col_center)

        return cv2.copyMakeBorder(cutted, top=top_pad, bottom=bottom_pad,
                                  left=left_pad, right=right_pad,
                                  borderType=cv2.BORDER_REFLECT)

    def set_image(self, image, base_name):
        assert isinstance(image, np.ndarray)
        self._im = image
        self._base_name = base_name
        self._save_count = 0

    def save_to_tar(self, filename, im):
        _, dec = cv2.imencode(".png", im)
        string = dec.tostring()
        byt = bytearray(string)
        memfile = io.BytesIO(byt)
        info = tarfile.TarInfo(filename)
        info.size = len(byt)
        self.tar.addfile(tarinfo=info, fileobj=memfile)

    def close_tar(self):
        if self.save_as_tar:
            self.tar.close()

    def purge_previous_im(self, tar_name):
        if self.save_as_tar:
            if os.path.isfile(tar_name):
                os.remove(tar_name)
        else:
            filter = ['*.bmp', '*.png', '*.tif']
            filelist = utils.listFiles(self._saveDir, filter)
            for fileinfo in filelist:
                im_path = fileinfo.absoluteFilePath()
                os.remove(im_path)


class No_save_ImCutter(ImCutter):
    def __init__(self, im, cut_size=63):
        self._im = im
        self._cut_size = int(cut_size / 2)

    def cut(self, row_center, col_center):
        assert isinstance(self._im, np.ndarray)
        left, right, top, bottom = self.get_borders(row_center, col_center)
        cut = self._im[top:bottom + 1, left:right + 1]
        return self.pad_image(cut, row_center, col_center)