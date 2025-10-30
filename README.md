# Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras

![alt text](static/intro.png)

This is the official repository for the paper: *Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras*.

Spiking Patches is implemented in Rust. See *src/* and *sp/tokenizer.py* if you are mainly interested in the code for Spiking Patches. The rest of the code is related to the experiments in the paper.

**Citation**

``` bibtex
@InProceedings{Ohrstrom_2025,
  author = {Øhrstrøm, Christoffer Koo and Güldenring, Ronja and Nalpantidis, Lazaros},
  title  = {Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras},
  year   = {2025},
}
```

## Getting Started

You will need [uv](https://docs.astral.sh/uv/), [Rust](https://rust-lang.org/), and [Python](https://www.python.org) to install the project.

We expect at least Python v3.12 and recommend installing it through uv.
The project has been tested with rustc v1.90.0.

Run the install script to install the project in a new virtual environment.

``` **sh**
sh install.sh
```

You should also create these directories at the project root:

``` txt
datasets/
experiments/
```

Datasets must be placed in *datasets* (see [Datasets](#datasets)).
Model checkpoints will automatically be placed in *experiments*.

We further log experiments (hyperparameters and learning curves) to [Weights & Biases](https://wandb.ai) and expect that your user has access to a project called "*spiking-patches*". Please create this project. You can change this in `sp/constants.py` if you want a different project name.

## Datasets

Download links to the datasets used for experiments:

| Dataset        | URL                                                                                              |
| :------------- | :----------------------------------------------------------------------------------------------- |
| DvsGesture     | [Download](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)        |
| GEN1           | [Download](https://www.prophesee.ai/2020/01/24/prophesee-gen1-**automotive**-detection-dataset/) |
| SL-Animals-DVS | [Download](http://www2.imse-cnm.csic.es/caviar/SL_Animals_Dataset/)                              |

The datasets should be unpacked in *datasets/*. For your convenience, we show the expected folder structure from the root of the dataset directory. Make sure that your unpacked datasets directory adhere to this structure:

``` txt
dvs_gesture/
    - DvsGesture/
gen1/
    - test/
    - train/
    - val/
sl-animals-dvs/
    - allusers_aedat/
    - tags_updated_19_08_2020/
```

## Preprocess

The datasets must be preprocessed before they are ready to be used for training and evaluation.

The commands below replicate the setup in our experiments:

``` sh
uv run preprocess.py dvsgesture
uv run preprocess.py slanimalsdvs
uv run preprocess.py gen1 --chunk-duration-ms 50
```

## Training

All datasets use the same entrypoint: *train.py*.

For example, you can train a Transformer (T) on DvsGesture (DG) with Spiking Patches (SP) like this:

``` sh
uv run train.py DG-T-SP
```

This will create a new experiment with a random name. Use the `--name` flag to give the experiment a custom name. You will refer to this name later when evaluating the model.

``` sh
uv run train.py DG-T-SP --name DG-T-SP
```

You can add the `--debug` flag to avoid saving checkpoints and logging to Weights & Biases.

``` sh
uv run train.py DG-T-SP --debug
```

Use the `--help` flag to see all options and their defaults.

``` sh
uv run train.py --help  # for all data/model/tokenizer setups
uv run train.py DG-T-SP --help  # for all configurations of the given setup
```

## Evaluation

Use *evaluate.py* to evaluate a model:

``` sh
uv run evaluate.py [[name]]
```

`[[name]]` refers to the name assigned in training.
The results are logged to Weights & Biases.

Evaluation defaults to use the validation split and the checkpoint that obtained the highest validation score during training. Use the `--split` and `--checkpoint` options to change this:

``` sh
uv run evaluate.py [[name]] --split test --checkpoint last
```

## Acknowledgments

This project has used code from the following projects:

* [Prophesee toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox): For loading data from Prophesee datasets (GEN1).
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX): For the object detection head.