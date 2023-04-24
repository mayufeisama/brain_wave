# Python example code for the George B. Moody PhysioNet Challenge 2023

## What's in this repository?

This repository contains a simple example that illustrates how to format a Python entry for the George B. Moody PhysioNet Challenge 2023. We recommend that you use this repository as a template for your entry. You can remove some of the code, reuse other code, and add new code to create your entry. You do not need to use the models, features, and/or libraries in this example for your approach. We encourage a diversity of approaches for the Challenge.

For this example, we implemented a random forest model with several features. This simple example is designed **not** not to perform well, so you should **not** use it as a baseline for your model's performance. You can try it by running the following commands on the Challenge training set. These commands should take a few minutes or less to run from start to finish on a recent personal computer.

This code uses four main scripts, described below, to train and run a model for the Challenge.

## How do I run these scripts?

You can install the dependencies for these scripts by creating a Docker image (see below) and running

    pip install -r requirements.txt

You can train your model by running

    python train_model.py training_data model

where

- `training_data` (input; required) is a folder with the training data files and
- `model` (output; required) is a folder for saving your model.

You can run your trained model by running

    python run_model.py model test_data test_outputs

where

- `model` (input; required) is a folder for loading your model,
- `test_data` (input; required) is a folder with the validation or test data files (you can use the training data for debugging and cross-validation, but the validation and test data will not have labels and will have 12, 24, 48, or 72 hours of data), and
- `test_outputs` is a folder for saving your model outputs.

The [Challenge website](https://physionetchallenges.org/2023/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2023) and running

    python evaluate_model.py labels outputs scores.csv

where

- `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage,
- `outputs` is a folder containing files with your model's outputs for the data, and
- `scores.csv` (optional) is a collection of scores for your model.

## Which scripts I can edit?

Please edit the following script to add your code:

* `team_code.py` is a script with functions for training and running your trained model.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model.
* `run_model.py` is a script for running your trained model.
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your models, please edit the `train_challenge_model` function in the `team_code.py` script. Please do not edit the input or output arguments of the `train_challenge_model` function.

To load and run your trained model, please edit the `load_challenge_model` and `run_challenge_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of the functions of the `load_challenge_model` and `run_challenge_model` functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments .

To guarantee that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2023/#data). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2023.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-example-2023  test_data  test_outputs  training_data

        user@computer:~/example$ cd python-example-2023/

        user@computer:~/example/python-example-2023$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2023$ docker run -it -v ~/example/model:/challenge/model -v ~/example/test_data:/challenge/test_data -v ~/example/test_outputs:/challenge/test_outputs -v ~/example/training_data:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py

        root@[...]:/challenge# python train_model.py training_data model

        root@[...]:/challenge# python run_model.py model test_data test_outputs

        root@[...]:/challenge# python evaluate_model.py test_data test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

## What else do I need?

This repository does not include code for evaluating your entry. Please see the [evaluation code repository](https://github.com/physionetchallenges/evaluation-2023) for code and instructions for evaluating your entry using the Challenge scoring metric.

This repository also includes code for preparing the validation and test sets. We will run your trained model on data without labels and with 12, 24, 48, and 72 hours of recording data to evaluate its performance with limited amounts of data. You can use this code to prepare the training data in the same way that we prepare the validation and test sets.

- `truncate_data.py`: Truncate the EEG recordings. Usage: run `python truncate_data.py -i input_folder -o output_folder -k 12` to truncate the EEG recordings to 12 hours. We will run your trained models on data with 12, 24, 48, and 72 hours of recording data.
- `remove_labels.py`: Remove the labels. Usage: run `python remove_labels.py -i input_folder -o output_folder` to copy the data and metadata (but not the labels) from `input_folder` to `output_folder`.
- `remove_data.py`: Remove the binary signal data, i.e., the EEG recordings. Usage: run `python remove_data.py -i input_folder -o output_folder` to copy the labels and metadata (but not the EEG recording data) from `input_folder` to `output_folder`.

## How do I learn more?

Please see the [Challenge website](https://physionetchallenges.org/2023/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [Challenge website](https://physionetchallenges.org/2023/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2023)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2023)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2023/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
