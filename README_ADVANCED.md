# NIPGBoard for Tensorboard

The NIPGBoard is a modified version of Tensorboard designed for the specific task of quickly annotating based on the positive or negative
association between object pairs, designed to allow annotation and training alike through ease of use.

## Installation Summary

* Establish a LOGDIR directory
* Download dependencies and run setup script
* Enter the virtual environment
* Run Step 0 preprocessing script, either ours or a custom one
* Create Step N training script or have ours ready
* Set configuration json file: set correct folder names, and parameters for a custom training script
* Build and run

## Installation for Ubuntu

### Setting up the "LOGDIR"

The term LOGDIR will be necessary throughout the installation and running of NIPGBoard. It refers to the **ABSOLUTE PATH** of an accessible directory that shall contain everything relevant to the application, which at start should be:

    LOGDIR
    ├── image_folder            # The folder containing the relevant database of *.png* images.
    │   ├── ...       
    │   ├── *some_image.png*   
    │   └── ...              
    └── ...

As an example, from the Downloads section aquire the zipped *traffic.zip* folder which includes the traffic signs dataset (subsample of https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and the corresponding necessary models
and metadata files. 

### Dependencies and Setup

In order to run the NIPGBoard, you need to first have the required dependencies:

 * Install Bazel version 0.26.1 through https://github.com/bazelbuild/bazel/releases/tag/0.26.1
 
 * Cuda 10.0 for GPU compatiblity

After that, having already set up the LOGDIR directory, you can install the remaining requirements and the virtual environment
with the install script using:

```
$ ./setup.sh LOGDIR
```

Where of course LOGDIR refers to the **ABSOLUTE PATH** of the directory containing the database and configuration file.

For any procedures onwards we recommend enterring the virtual environment generated, via:

```
$ source kira_venv/bin/activate
```

## Build and Run for Ubuntu

Be sure to enter the virtual environment and then execute from within the repository workspace via:

```
$ source kira_venv/bin/activate

$ bazel run tensorboard -- --logdir=LOGDIR
```

Where LOGDIR is the **ABSOLUTE PATH** directory containing the dataset.

When Bazel has finished building you can access the NIPGBoard via the URL displayed in the terminal, which by default is:

```
localhost:6006/#projector
```

Disclaimer: NIPGBoard has been developed and tested on Ubuntu 18.04

## Setting up prerequisites: Scripts and Data

### Explaining the Steps

The use-case for NIPGBoard follows the following pipeline:

 * We designate an accessible directory as the hub for all data, referred to as <LOGDIR>
 * We collect all our image data into a folder directly within the <LOGDIR>, referred to as <image_folder>
 * **Step 0:** In this step, before running the NIPGBoard, we perform a feature extraction/mapping on the contents of <image_folder>. This allows us to visualize a starting point from which the user will begin annotating positive and negative relations between pairs. An example for this is provided in the *algorithms/vgg.py* script that generates a feature extraction using VGG16.
 * **Step N:** Having already built and ran the NIPGBoard, it will now use our training script, and the Step N-1 results to generate a new clustering based on the annotated pairs. An example for this is provided in the *algorithms/kira.py* script that fine-tunes using the pairs.

 **DISCLAIMER: under poor network conditions you might have to refresh the NIPGBoard site more than once to load the Step N embedding results.**

### Example scripts: feature extraction, training, testing mockups

We provide a script in **algorithms/vgg.py** that executes **Step 0**, by doing feature extractions for the given image dataset using VGG16's model pretrained on the imagenet database. This allows the user to see a relevant starting clustering from which to run the pair annotation from.
Execute it before passing the NIPGBoard to a user; as a preprocessing procedure, via:

```
$ source kira_venv/bin/activate
$ cd ./algorithms
$ python vgg.py <logdir> <image_folder> <embedding_folder>

#for example:
$ python vgg.py /home/ervin/logdir/ traffic_images traffic_vgg
```

* *<logdir>*: Refers to the absolute path to LOGDIR
* *<image_folder>*: Refers to the folder name within LOGDIR where the image database is located
* *<embedding_folder>*: Refers to the folder name where the feature extraction results will be saved. 

A mockup version exists as well for testing purposes for **Step 0** in the script **algorithms/mockup_step_0.py**, which generates a randomized but functional embedding via the command:

```
$ source kira_venv/bin/activate
$ cd ./algorithms
$ python mockup_step_0.py <logdir> <image_folder> <embedding_folder>

#for example:
$ python mockup_step_0.py /home/ervin/logdir/ traffic_images traffic_vgg
```

The parameters are equivalent to vgg.py.

We provide a script in **algorithms/kira.py** that executes **Step N**, by fine-tuning the VGG16 model using a specialized replacement top-level layer and utilizing the pairs annotated in NIPGBoard. This will be executed via the NIPGBoard and the **<logdir>/cnf.json** file should be set according to the training algorithm settings. We provide a default configuration that sets this script's parameters in the configuration file, so you only have to edit the folder names.

This also has a randomized mockup version in **algorithms/mockup_step_N.py** for parsing the NIPGBoard call parameters but still randomizing a training result.

### Generating the embedding for NIPGBoard

As long as the algorithm has access to the base directory of the NIPGBoard repository / workspace, either by existing directly in it or
via adding it to the Pythonpath; the developer can import and use the following method in the python script to create a displayable embedding, which be used in both Step 0 and Step N to generate the visualizations.
**Warning: if outside of the virtual environment generated during installation, be careful of any other versions of Tensorboard installed that do not include the custom helper functions.**

```
from tensorboard.plugins.executer.executer_plugin import makeProjector
makeProjector(data, name, logdir, image_folder, embedding_folder)
```

 * *data:* Numpy array containing the tensor.
 * *name:* Tensor name
 * *logdir:* LOGDIR absolute path
 * *image_folder:* Folder directly inside LOGDIR containing the image database
 * *embedding_folder:* Folder directly inside LOGDIR where the embedding will be saved

One can also import and use the sprite creation method for generating the images renderred in the 3D area within the Projector, albeit if
omitted, the NIPGBoard will take care of this itself.

```
from tensorboard.plugins.executer.executer_plugin import imageTensorToSprite
imageTensorToSprite(data, imagePath)
```

 * *data:* Tensor containing the images.
 * *imagePath:* LOGDIR + image_folder.

### Parameters for the Training Script

For the "Step N" training script to be called properly from within the NIPGBoard, it must have a callable method with the following signature:

```
def run(logdir, boardPath, imagePath, pairs, embeddingPath, ...*args*..., ...*kwargs*...)
```

 * The NIPGBoard will pass the LOGDIR absolute path to the script via *logdir*
 * The NIPGBoard will pass it's own repository absolute path to the script via *boardPath*
 * The NIPGBoard will pass the <image_folder> name directly within LOGDIR via *imagePath*
 * The NIPGBoard will pass a numpy array containing the positive/negative pair labels shaped *number_of_images\*number_of_images* via *pairs*
 * The NIPGBoard will pass the name of the folder directly within LOGDIR where the embedding will be saved via *embeddingPath*
 * The NIPGBoard will pass any additional arguments and keyword arguments that are provided.
 * **The training script should be using these parameters and pairs to create the new embedding to be shown in NIPGBoard**

This is customized in the section below using the configuration file.

### Setting up the configuration file.

Upon finishing the installation it will automatically generate an example configuration file into LOGDIR called *cnf.json*. It should look like
this:

```
{
	"default": {                                          # This contains the "Step 0" feature information.
		"name": "VGG",                                    # A name given to the "Step 0" feature extraction embedding.
		"embedding_folder": "traffic_vgg",                      # The input folder name directly within the LOGDIR containing the "Step 0" embedding.
		"image_folder": "traffic_images"                  # The input folder name directly within the LOGDIR containing the image database.
	},

	"training": {                                                   # This containts the "Step N" training information.
		"name": "KIRA",                                             # A name given to the "Step N" feature extraction embedding.
		"embedding_folder": "traffic_kira",                                 # The folder name where training embeddings are saved / loaded.
		"algorithm_path": "/home/ervin/nipg-board-v3/algorithms",   # The absolute path where the training script python module can be found.
		"algorithm": {                                              # This contains the information for the training module.
			"file": "kira",                                                     # The name of the python module.
			"callable": "run",                                                  # The callable within the module that initiates the training.
			"arguments": [],                                                    # Arguments for the callable.
			"keyword_arguments": {"epoch": "10"}                                # Keyword arguments for the callable.
		}
	}
}
```

**Upon installing NIPGBoard we will auto-generate a version of *cnf.json* that is only missing the image and embedding folder names, and is otherwise configured for the algorithms/kira.py training script provided. Please be sure to replace the placeholder image and embedding folder names.**

Thus, when using the NIPGBoard, the LOGDIR should effectively look like this:

    LOGDIR
    ├── image_folder                # The folder containing the relevant database of *.png* images.
    │   ├── ...       
    │   ├── *some_image.png*   
    │   └── ...              
    ├── traffic_vgg    # The folder containing the "Step 0" feature extraction embedding.
    │   └── ...              
    ├── traffic_kira   # The folder containing (or that will contain) the "Step N" feature extraction embedding.
    │   └── ...              
    |   cnf.json                    # Configuration file.
    └── ...