# -*- coding: utf-8 -*-

"""

Copyright (c) 2016 by Patrick Hall, jpatrickhall@gmail.com

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

Python script and raw images to generate a labeled image data set.

Images are copied into n_process (default 2) seperate folders and converted to 
greyscale. Images are then tiled into smaller, square patches. After tiles are 
created they are the tested for pixel intensity variance to prevent nearly 
constant patches from entering the generated data set. Patches with sufficently
varying pixel intensity values are then down- or up- sampled to a standard size
(default 25 x 25). Users may also specify the -a (default 5) option or set the 
angle constant in main to create additional rotated copies of each patch. All 
patches, including any rotated copies are then flattened and collected into a 
single csv file, called 'patches.csv'. Patches.csv is located in out_dir. 
Patches.csv will contain all pixel intensity values for each patch as a row 
vector. Each row of patches.csv will also contain an original image id along 
with the upper-lefthand x and y values of the patch in the original image and 
the size, angle, and label of the patch. Patches are labeled as containing 
Frank (the cat) or not. All pre-existing files from earlier runs of 
threaded_tile_r.py may be replaced or deleted by subsequent runs.

Important constants:

n_process: (-p) Number of processes to use; the script will create this many
           chunks of image files and place them into working sub-directores,
           with names like out_dir/_chunk_dir<n>. (default=2)

in_dir: (-i) Directory in which input images are located. Files will be copied 
        into their respective chunk directories before being tiled, (and/or 
        downsampled,) flattened, and converted to csv. (default='./in')

out_dir: (-o) Parent directory in which the sub-directories for each chunk of
         image files will be created. A large number temporary files will be
         created in out_dir. Patches.csv is located in out_dir. 
         (default='./out)

debug: (-g) Leaves temporary files in out_dir. (default=True)

downsample_size: (-d) Side length of downsampled square image patches measured
                 in pixels. By default image patches are downsampled before
                 being flattened, resulting in each row vector of patches.csv
                 containing downsample_size*downsample_size pixels intensity
                 elements. (default=25)

variance_threshold: (-v) The standard deviation above which an image will be
                    flattened and saved to patches.csv. Used to prevent blank,
                    dark, or plain white images from being included in the
                    generated data set. (default is dependent on image size)

angle: (-a) Patches will be rotated this angle in degrees before being
       flattened and added to patches.csv. Patches are alternately rotated in
       positive and negative directions to prevent creating many copies of each
       patch. (default=5)

Run threaded_tile_r.py from an IDE by setting constants in main OR by the
command line. Example command line usage:

$ python threaded_tile_r.py -i in -o out -v 55

"""

# imports

import ast
import csv
import getopt
import multiprocessing
import numpy as np
import os
import shutil
import sys
import time
from multiprocessing import Process
from PIL import Image

def create_out_dirs(n_process, out_dir):

    """ Creates n_process number of output directories.

    Args:
        n_process: Number of processes specified by the user.
        out_dir: Directory in which to create intermediate files and final
                 patches.csv file.

    Raises:
        EnvironemtError: Problem creating directories.
    """

    print '-------------------------------------------------------------------'
    print 'Creating working directory structure ... '

    # create dir structure

    for i in range(0, int(n_process)):

        chunk_outdir = out_dir + os.sep + '_chunk_dir' + str(i)
        try:
            if os.path.exists(chunk_outdir):
                shutil.rmtree(chunk_outdir, ignore_errors=True)
            os.mkdir(chunk_outdir)
            print 'Created ' + chunk_outdir + ' ...'
        except EnvironmentError as exception_:
            print exception_
            print 'Failed to locate or create ' + chunk_outdir + '!'
            sys.exit(-1)

    print 'Done.'

def chunk_files(n_process, in_dir, out_dir):

    """ Separates the image in in_dir into n_process roughly equal chunks of
    files, each in a separate directory created by create_out_dirs.

    Args:
        n_process: Number of processes specified by the user.
        in_dir: Directory in which original image files are located.
        out_dir: Directory in which to create intermediate files and final
                 patches.csv file.

    Raises:
        EnvironemtError: Problem copying image files.
    """

    print '-------------------------------------------------------------------'
    print 'Chunking ' + in_dir + ' ...'

    # local constants

    check_point_value = 1000
    image_type_list = ['JPG', 'JPEG', 'PNG', 'BMP', 'TIFF']

     # copy files

    file_list = [name for name in os.listdir(in_dir)\
                    if name.split('.')[-1].upper() in image_type_list]

    for i, name in enumerate(file_list):

        source_file = in_dir + os.sep + name
        chunk_outdir = out_dir + os.sep + '_chunk_dir' +\
            str(int(i % n_process))
        chunk_file = chunk_outdir + os.sep + name

        try:
            if os.path.isfile(chunk_file):
                shutil.rmtree(chunk_file, ignore_errors=True)
            else:
                shutil.copy(source_file, chunk_file)
        except EnvironmentError as exception_:
            print exception_
            print 'Failed to copy' + name + '!'
            sys.exit(-1)

        if i % check_point_value == 0 and i != 0:
            print 'Processing file %i ...' % (i)

    print 'Done.'

def map_make_tiles(i, out_dir, debug, downsample_size, variance_threshold,
                   angle):

    """ In each process: by default creates differently sized patches,
    conditionally creates rotated copies of the patches, tests patches for
    sufficient pixel intensity variance, down-or up-samples the
    patches, flattens patches into row vector of pixel intensities, and saves
    row vector to intermediate csv. Variance threshold and rotation angle
    defaults can be overridden by setting constants in main or by command line
    options.

    Args:
        i: Process index.
        out_dir: Directory in which to create intermediate files and final
                 patches.csv file.
        debug: If true, preserves intermediate image patches.
        downsample_size: Side length of downsampled square image patches 
                         measured in pixels. By default, image patches are
                         downsampled before being flattened.
        variance_threshold: The standard deviation above which an image will be
                            flattened and saved to patches.csv.
        angle: patches will be rotated this angle in degrees before being
               flattened and added to patches.csv.

    Raises:
        EnvironemtError: Problem creating csv file.
    """

    # local constants

    process_name = multiprocessing.current_process().name
    chunk_dir = out_dir + os.sep + '_chunk_dir' + str(i)
    image_type_list = ['JPG', 'JPEG', 'PNG', 'BMP', 'TIFF']
    tiles_per_short_side = 100 # increase to create more patches 
    min_size = 250 # decrease to create a more noisy classification problem

    # open intermediate csv for writing flattened images

    out_csv_name = chunk_dir + os.sep + 'patches' + str(i) + '.csv'
    try:
        if os.path.exists(out_csv_name):
            shutil.rmtree(out_csv_name, ignore_errors=True)
        o = open(out_csv_name, 'wb')
        wr = csv.writer(o)
    except EnvironmentError as exception_:
        print exception_
        print 'Failed to create ' + out_csv_name + '!'
        sys.exit(-1)

    file_list = [name for name in os.listdir(chunk_dir)\
                    if name.split('.')[-1].upper() in image_type_list]

    for name in file_list:

        print process_name + ': tiling ' + name + ' ...'

        chunk_file = chunk_dir + os.sep + name
        im = Image.open(chunk_file).convert('L')
        w, h = im.size
        short_side_length = min(w, h)

        # assign label
        if name.upper().find('NOT') >= 0:
            label = 0
        else:
            label = 1

        # check variance_threshold
        if variance_threshold == None:
            variance_threshold = short_side_length/60

        # init stride_length based on image size
        stride_length = int(short_side_length/tiles_per_short_side)

        # init tile_counter
        tile_counter = 0

        # init size_
        np.random.seed(1234)
        size_ = np.random.randint(min_size, short_side_length)
        
        # create patches from each file #######################################

        reached_y_edge = False
        for y in range(0, h, stride_length):

            y_ = y
            if reached_y_edge:
                continue
            else:
                my = min(y + size_, h)
                if my == h:
                    y_ = h - size_
                    reached_y_edge = True

            reached_x_edge = False
            for x in range(0, w, stride_length):

                x_ = x
                if reached_x_edge:
                    continue
                else:
                    mx = min(x + size_, w)
                    if mx == w:
                        x_ = w - size_
                        reached_x_edge = True

                tile = im.crop((x_, y_, mx, my))

                # conditionally rotate every other image
                # and over write patch
                angle_ = 0
                if angle != None and angle != 0:
                    im_rplus = im.rotate(angle)
                    im_rminus = im.rotate(-angle)
                    if x > 0 and not reached_x_edge:
                        if y > 0 and not reached_x_edge:
                            if tile_counter % 2 == 0:
                                if tile_counter % 4 == 0:
                                    angle_ = angle
                                    tile = im_rplus.crop((x_, y_, mx, my))
                                else:
                                    angle_ = -angle
                                    tile = im_rminus.crop((x_, y_, mx, my))

                # check image variance
                std = np.std(np.array(tile))
                if std > variance_threshold:

                    # conditionally downsample patches
                    if downsample_size != None:
                        tile = tile.resize((downsample_size, downsample_size),\
                                            Image.ANTIALIAS)

                    if debug:
                        tile_fname = os.path.join(chunk_dir,\
                            'patch.%s.%d.%d.%d.%d.png' %\
                            (name, x_, y_, size_, angle_))
                        tile.save(tile_fname, "PNG")

                    # flatten patches into row vector of pixel intensities
                    tile_list = list(tile.getdata())

                    #  add tile attributes, including label
                    tile_list.extend([name, x_, y_, size_, angle_, label])

                    # save row vector to intermediate csv
                    wr.writerow(tile_list)

                # init next iter
                tile_counter +=1
                size_ = np.random.randint(min_size, short_side_length)

    o.close()
    print process_name + ': Done.'

def reduce_join_tile_csv(n_process, out_dir, debug, tile_size):

    """ Creates out_dir/patches.csv, writes csv header to patches.csv, and
    concatenates intermediate csv files into patches.csv.

    Args:
        n_process: Number of processes specified by the user.
        out_dir: Directory in which to create intermediate files and final
                 patches.csv file.
        debug: If true, preserves intermediate working directories.
        tile_size: Side length of down- or up- sampled square image patches
                   measured in pixels - used to create csv header here.

    Raises:
        EnvironemtError: Problem creating csv file.
    """

    # create out_dir/patches.csv

    out_csv_name = out_dir + os.sep + 'patches.csv'
    try:
        if os.path.exists(out_csv_name):
            shutil.rmtree(out_csv_name, ignore_errors=True)
        o = open(out_csv_name, 'wb')
    except EnvironmentError as exception_:
        print exception_
        print 'Failed to create ' + out_csv_name + '!'
        sys.exit(-1)

    # write csv header to patches.csv

    header = ['pixel_' + str(j) for j in range(0, tile_size*tile_size)]
    header.extend(['orig_name', 'x', 'y', 'size', 'angle', 'label'])
    csv.writer(o).writerow(header)

    # concatenate intermediate csv files into patches.csv

    for i in range(0, n_process):
        chunk_dir = out_dir + os.sep + '_chunk_dir' + str(i)
        in_csv_name = chunk_dir + os.sep + 'patches' + str(i) + '.csv'
        with open(in_csv_name) as n:
            for line in n:
                o.write(line)
        if not debug:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    o.close()

def main(argv):

    """ For running standalone.
    Args:
        argv: Command line args.
    Raises:
        GetoptError: Problem parsing command line options.
        BaseException: Some problem from a multiprocessing task.
    """

    # TODO: user set constants if running from IDE
    # init local vars to defaults

    n_process = 2
    in_dir = './in'
    out_dir = './out'
    debug = True
    downsample_size = 25
    variance_threshold = None
    angle = 5

    # parse command line args and update dependent args

    try:
        opts, _ = getopt.getopt(argv, "p:i:o:g:d:v:a:h")
        for opt, arg in opts:
            if opt == '-p':
                n_process = int(arg)
            elif opt == '-i':
                in_dir = arg
            elif opt == '-o':
                out_dir = arg
            elif opt == '-g':
                debug = ast.literal_eval(arg)
            elif opt == '-d':
                downsample_size = int(arg)
            elif opt == '-v':
                variance_threshold = int(arg)
            elif opt == '-a':
                angle = int(arg)
            elif opt == '-h':
                print 'Example usage: python threaded_tile.py \
-i <input directory> -o <output directory> -v <variance threshold>'
                sys.exit(0)
    except getopt.GetoptError as exception_:
        print exception_
        print 'Example usage: python threaded_tile.py\
-i <input directory> -o <output directory> -v <variance threshold>'
        sys.exit(-1)

    print '-------------------------------------------------------------------'
    print 'Proceeding with options: '
    print 'Processes (-p)           = %s' % (n_process)
    print 'Input directory (i)      = %s' % (in_dir)
    print 'Output directory (-o)    = %s' % (out_dir)
    print 'Debug (-g)               = %s' % (debug)
    print 'Downsample size (-d)     = %s' % (downsample_size)
    print 'Variance threshold(-v)   = %s' % (variance_threshold)
    print 'Angle (-a)               = %s' % (angle)

    # start execution timer

    bigtic = time.time()

    # init chunk directory structure and copy chunks of files

    create_out_dirs(n_process, out_dir)
    chunk_files(n_process, in_dir, out_dir)

    # multiprocessing map/reduce scheme to execute image manipulation tasks on
    # chunks of image files in parallel

    # tile images using multiprocessing
    # store in temporary files

    print '-------------------------------------------------------------------'
    print 'Tiling images ... '
    tic = time.time()
    processes = []
    try:
        for i in range(0, int(n_process)):
            process_name = 'Process_' + str(i)
            process = Process(target=map_make_tiles, name=process_name,\
            args=(i, out_dir, debug, downsample_size,\
                  variance_threshold, angle))
            process.start()
            processes.append(process)
        for process_ in processes:
            process_.join()
        print 'Images tiled in %.2f s.' % (time.time()-tic)
    except BaseException as exception_:
        print exception_
        print 'ERROR: Could not tile images.'
        print sys.exc_info()
        exit(-1)

    # reduce temporary files into a single large csv

    print '-------------------------------------------------------------------'
    print 'Combining tile csv files ... '
    tic = time.time()
    reduce_join_tile_csv(n_process, out_dir, debug, downsample_size)
    print 'Done.'
    print 'Csv files combined in %.2f s.' % (time.time()-tic)

    print '-------------------------------------------------------------------'
    print 'All tasks completed in %.2f s.' % (time.time()-bigtic)

if __name__ == '__main__':
    main(sys.argv[1:])
