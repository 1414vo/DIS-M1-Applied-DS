# Applied Data Science - ivp24

This repository contains a solution for the M1 Applied Data Science coursework.

![Static Badge](https://img.shields.io/badge/build-passing-lime)
![Static Badge](https://img.shields.io/badge/logo-gitlab-blue?logo=gitlab)

## Table of contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the tasks](#running-the-tasks)
5. [Frameworks](#frameworks)
6. [Build status](#build-status)
7. [Credits](#credits)

## Requirements

The user should preferrably have a version of Docker installed in order to ensure correct setup of environments. If that is not possible, the user is recommended to have Conda installed, in order to set up the requirements. If Conda is also not available, make sure that the packages described in `environment.yml` are available and installed.

## Setup

We provide two different set up mechanisms using either Docker or Conda. The former is recommended, as it ensures that the environment used is identical to the one used in the development in the project.

### Using Docker

To correctly set up the environment, we utilise a Docker image. To build the image before creating the container, you can run.

```docker build -t ivp24_ads .```

The setup image will also add the necessary pre-commit checks to your git repository, ensuring the commits work correctly. You need to have the repository cloned beforehand, otherwise no files will be into the working directory.

Afterwards, any time you want to use the code, you can launch a Docker container using:

```docker run --name <name> --rm -p 8888:8888 -ti ivp24_ads```

If you want to make changes to the repository, you would likely need to use your Git credentials. A safe way to load your SSH keys was to use the following command:

```docker run --name <name> --rm -p 8888:8888 -v <ssh folder on local machine>:/root/.ssh -ti ivp24_ads```

This copies your keys to the created container and you should be able to run all required git commands.

### Using Conda

The primary concern when using Conda is to install the required packages. In this case **make sure to specify an environment name**. Otherwise, you risk overriding your base environment. Installation and activation can be done using the commands:

```conda env create --name <envname> -f environment.yml ```
```conda activate <envname> ```

## Running the tasks

Each task has been solved in a separate Jupyter Notebook, found in the `/notebooks` folder. Notebooks were chosen instead of executables due to the vast amount of plots that needed to be presented.

To start Jupyter, simply run one of the following commands:
- From a Docker container: ```jupyter notebook --ip 0.0.0.0 --no-browser```
- From a local terminal ```jupyter notebok --ip=*```

The notebooks can then be executed sequentially. Note that Question 4 takes a long amount of time to run (15-20 minutes if the machine is overloaded).
## Frameworks

The entire project was built on **Python** and uses the following packages:
- For computation and data processing:
    - NumPy
    - Pandas
- Machine Learning:
    - Scipy
    - Scikit-learn
- For plotting:
    - matplotlib
- For maintainability/documentation:
    - doxygen
    - pytest
    - pre-commit

## Build status
Currently, the build is complete and the program can be used to its full capacity.

## Credits

Plotting utilities were adapted from <a href="https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489">TowardsDataScience</a>.
