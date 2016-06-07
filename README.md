# frank_not_frank

Python script and raw images to generate a labeled data set.

Generated images are labeled as containing Frank (the cat) or not.

Licensed for commercial use.

## License

Copyright (c) 2015 by Patrick Hall, jpatrickhall@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
See the License for the specific language governing permissions and 
limitations under the License.

## Instructions

### Install required software

Git client - https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Git LFS client - https://git-lfs.github.com/

Anaconda Python - https://www.continuum.io/downloads

### Fork and pull materials

Fork the frank_not_frank repository from https://github.com/jphall663/frank_not_frank.git

Open a git bash terminal and enter the following statements on the git bash command line:

`$ mkdir frank_not_frank`

`$ cd frank_not_frank`

`$ git init`

`$ git lfs install`

`$ git remote add origin https://github.com/<your username>/frank_not_frank.git`

`$ git remote add upstream https://github.com/jphall663/frank_not_frank.git`

`$ git pull origin master`

### Generate the image data set

`$ python threaded_tile_r.py`

Note that the threaded_tile_r.py script has several options that can be set within the main function of the script or invoked from the command line:

#### n_process (-p)

Number of processes to use; the script will create this many chunks of image files and place them into working sub-directores, with names like out_dir/_chunk_dir<n>. (default=2)

#### in_dir (-i)

Directory in which input images are located. Files will be copied into their respective chunk directories before being tiled, (and/or downsampled,) flattened, and converted to csv. (default='./in')

#### out_dir: (-o)

Parent directory in which the sub-directories for each chunk of image files will be created. A large number of temporary files will be created in out_dir. Patches.csv is located in out_dir. (default='./out)

#### debug (-g)

Leaves temporary files in out_dir. Also useful for generating images files to read directly. (default=True)

#### downsample_size (-d)

Side length of downsampled square image patches measured in pixels. By default image patches are downsampled before being flattened, resulting in each row vector of patches.csv containing downsample_size*downsample_size pixels intensity elements. (default=25)

#### variance_threshold (-v)

The standard deviation above which an image will be flattened and saved to patches.csv. Used to prevent blank, dark, or plain white images from being included in the generated data set. (default is dependent on image size)

#### angle (-a)

Patches will be rotated this angle in degrees before being flattened and added to patches.csv. Patches are alternately rotated in positive and negative directions to prevent creating many copies of each patch. (default=5)