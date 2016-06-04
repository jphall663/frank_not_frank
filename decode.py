# -*- coding: utf-8 -*-
"""

Copyright (c) 2016 by Patrick Hall
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------

@author: jpatrickhall@gmail.com

"""

import csv
import os
import getopt
import sys
import time
from multiprocessing import Process
from PIL import Image
from random import randint

def print_progress(iter_, total, prefix=''):

    """ Utility function to print progress bar.

    Args:
        iter_:  current iteration
        total:  total iterations
        prefix: progress bar message prefix
    """

    suffix = 'complete' # progress bar message default suffix
    decimals = 1        # number of places to display after decimal
    bar_len = 50        # total length of progress bar

    # update progress bar
    filled_len = int(round(bar_len * iter_ / float(total)))
    percents = round(100.0 * (iter_ / float(total)), decimals)
    prog_bar = '#' * filled_len + '-' * (bar_len - filled_len)

    # print/reprint progress bar
    # /r is cariage return - allows reprint
    _ = sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, prog_bar, percents,\
                                                '%', suffix)),

    # move to next line at last iter
    sys.stdout.flush()
    if iter_ == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def rr_assign_files(n_process, dir_):

    """ Assign each file in dir_ to n_process different lists in round-robin
    way.

    Args:
        n_process: number of processes
        dir_:      dir to parse
    """

    # create list of all JPGs
    img_list = [fname for fname in os.listdir(dir_)\
                if fname.split('.')[-1].upper() in 'JPG']

    # init empty nested list for each process
    process_img_list = [[] for i in range(n_process)]

    # assign images to nested lists
    fname = ''
    for i, img_name in enumerate(img_list):
        fname = dir_ + os.sep + img_name
        process_img_list[int((i % n_process))].append(fname)

    return process_img_list

def read_labels(label_file):

    """ Read label file into a dictionary: key=image name, value=label.

    Args:
       label_file: path to label file
    """

    label_dict = {}
    with open(label_file, 'rb') as lfile:
        rdr = csv.reader(lfile)
        rdr.next()
        for line in rdr:
            label_dict[line[2]] = line[1]

    return label_dict

def map_downsample(i, img_dir, process_img_list, xy_pixels, label_dict=None):

    """ Decode images from process_img_list to greyscale, downsample, flatten
    and save to temporary csv; embarrasingly parallel, can be run in many
    processes concurrently.

    Args:
        i:                process number
        img_dir:          dir in which images are stored
        process_img_list: images from img_dir for this process
        xy_pixels:        list containing size of downsampled image [x,y]
        label_dict:       dictionary of key=image name, value=label
    """

    # open intermediate csv for writing flattened images
    out_csv_name = img_dir + os.sep + 'images' + str(i) + '.csv'
    if os.path.exists(out_csv_name):
        os.remove(out_csv_name)
    outf = open(out_csv_name, 'wb')
    wrtr = csv.writer(outf)

    # use first process to monitor progress
    # first process always present
    if i == 0:
        prefix = (img_dir.split(os.sep)[-1] + ':').ljust(9)
        print_progress(0, len(process_img_list) - 1, prefix)

    for j, name in enumerate(process_img_list):

        # convert to greyscale
        # downsample
        tile = Image.open(name).convert('L').resize((int(xy_pixels[0]),\
                                                     int(xy_pixels[1])),
                                                    Image.ANTIALIAS)

        # flatten tiles into row vector of pixel intensities
        tile_list = list(tile.getdata())

        # if this is training data, add labels and folder indicator
        if label_dict != None:
            # append label
            tile_list.append(label_dict[name.split(os.sep)[-1]])
            # append fold, random 1-5
            tile_list.append(randint(1, 5))

        # save row vector to intermediate csv
        wrtr.writerow(tile_list)

        # update progress
        if i == 0:
            print_progress(j, len(process_img_list) - 1, prefix)

    outf.close()


def reduce_join_csv(n_process, img_dir, xy_pixels, dir_list, role='test'):

    """ Collects temporary csvs created by map_downsample into single large
    csv.

    Args:
        n_process: number of processes
        img_dir:   dir in which images are stored
        xy_pixels: list containing size of downsampled images [x,y]
        dir_list:  list of directories in which to find temp csvs
        role:      'train' and 'test'; have different headers
    """

    # open final, large
    out_csv_name = img_dir + os.sep + str(role) + '.csv'
    if os.path.exists(out_csv_name):
        os.remove(out_csv_name)
    outf = open(out_csv_name, 'wb')

    # write csv header
    header = ['pixel_' + str(j) for j in range(0, int(xy_pixels[0])*\
               int(xy_pixels[1]))]
    if role == 'train':
        header.append('label')
        header.append('fold')
    csv.writer(outf).writerow(header)

    # concatenate intermediate csv files
    for dir_ in dir_list:
        for i in range(0, n_process):
            chunk_dir = img_dir + os.sep + dir_
            in_csv_name = chunk_dir + os.sep + 'images' + str(i) + '.csv'
            with open(in_csv_name) as nfile:
                for line in nfile:
                    outf.write(line)
            os.remove(in_csv_name)

    outf.close()

def main(argv):

    """ Run from IDE by setting constants below or run from command line.

    Args:
        argv: command line args
    """

    ### set constants here to run from IDE
    # n_process: number of processes to use when decodeing images
    # xy_pixel: list containing size of downsampled images [x,y]
    # data_dir: dir containing train, test img dirs and driver_imgs_list.csv
    n_process = 2
    xy_pixels = [40, 30]
    data_dir = ''

    # parse command line args and update dependent args

    try:
        opts, _ = getopt.getopt(argv, "p:x:y:d:h")
        for opt, arg in opts:
            if opt == '-p':
                n_process = int(arg)
            elif opt == '-x':
                xy_pixels[0] = int(arg)
            elif opt == '-y':
                xy_pixels[1] = int(arg)
            elif opt == '-d':
                data_dir = str(arg)
            elif opt == '-h':
                print 'Example usage: python decode.py \
-p <number of processes> \
-x <x size of downsampled images> \
-y <y size of downsampled images> \
-d <data directory>'
                sys.exit(0)
    except getopt.GetoptError as exception_:
        print exception_
        print 'Example usage: python decode.py \
-p <number of processes> \
-x <x size of downsampled images> \
-y <y size of downsampled images> \
-d <data directory>'
        sys.exit(-1)

    print '==================================================================='
    print 'Decoding images with options: '
    print 'Processes (-p)      = %s' % (n_process)
    print 'X size (-x)         = %s' % (xy_pixels[0])
    print 'Y size (-y)         = %s' % (xy_pixels[1])
    print 'Data directory (-d) = %s' % (data_dir)

    # start execution timer
    tic_ = time.time()

    print '==================================================================='

    ### train images

    # default subdirectory structure
    train_dir_list = ['train/c0', 'train/c1', 'train/c2', 'train/c3',\
                      'train/c4', 'train/c5', 'train/c6', 'train/c7',\
                      'train/c8', 'train/c9']

    # load labels
    label_dict = read_labels(data_dir + os.sep + 'driver_imgs_list.csv')

    for sub_dir in train_dir_list:

        # assign images to processes
        img_dir = data_dir + os.sep + sub_dir
        process_img_list = rr_assign_files(n_process, img_dir)

        # use multiprocessing to decode train images
        processes = []
        for i in range(0, int(n_process)):
            process = Process(target=map_downsample,\
            args=(i, img_dir, process_img_list[i], xy_pixels, label_dict,))
            process.start()
            processes.append(process)
        for process_ in processes:
            process_.join()

    # reduce temporary files into a single large csv
    reduce_join_csv(n_process, data_dir, xy_pixels, train_dir_list, 'train')
    sub_dir = 'test'

    ### test images

    # assign images to processes
    img_dir = data_dir + os.sep + sub_dir
    process_file_list = rr_assign_files(n_process, img_dir)

    # use multiprocessing to decode test images
    processes = []
    for i in range(0, int(n_process)):
        process = Process(target=map_downsample,\
        args=(i, img_dir, process_file_list[i], xy_pixels,))
        process.start()
        processes.append(process)
    for process_ in processes:
        process_.join()

    # reduce temporary files into a single large csv
    reduce_join_csv(n_process, data_dir, xy_pixels, ['test'])

    print '==================================================================='
    print 'Decoding completed in %.2f s.' % (time.time()-tic_)

if __name__ == '__main__':
    main(sys.argv[1:])



