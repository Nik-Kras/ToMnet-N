# Introduction: ToMnet-N

ToMnet-N (Theory of Mind network by Nikita) is a Deep Learning model of ToMnet family (ToMnet, ToM2C, Trait-ToM, ToMnet+). It is a Transformers version of ToMnet that is supposed to perform Theory of Mind tasks in a Grid World game. It observes bechaviour of the bot-player and performs predictions of trajectory, which type of agent it is and which goal a player is going to consume by the end of the game. 

# Methodology

- About the game
- About the AI player
- About ToMnet-N

![Example of a Maze #1](Results/Initial_Map.png)

# Project Structure

Project structure is based on Cookiecutter 
Please read here for description and specification on each folder: http://drivendata.github.io/cookiecutter-data-science/ 

```bash
├── LICENSE
├── Makefile            <- Makefile with commands like `make data` or `make train`
├── README.md           <- The top-level README for developers using this project.
├── data                <- Generated datasets
│
├── docs                <- A default Sphinx project; see sphinx-doc.org for details
│
├── models              <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks           <- Jupyter notebooks for model / data exploration. Naming convention is a number (for ordering),
│                          the creator\'s initials, and a short `-` delimited description, e.g.
│                          `1.0-jqp-initial-data-exploration`.
│
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
│
├── requirements.txt    <- The requirements file for reproducing the analysis environment
│
├── environment.yml     <- Environmental dependencies for reproducing the analysis environment
│
├── src                 <- Source code for use in this project.
│   ├── __init__.py     <- Makes src a Python module
│   │
│   ├── map_generation  <- Module to Cretae Maps with Wafe Function Collapse [Create Environment]
│   │   ├── tests       <- Unit tests for functionality inside `utils`
│   │   ├── utils       <- Helper functions and classes that implements Wave Function Collapse generation
│   │   └── map.py      <- Highly abstract API for map generation
│   │
│   ├── game_generation <- Module to Cretae Maps with Wafe Function Collapse [Record how player plays inside of Environment]
│   │   ├── tests       <- Unit tests for functionality inside `utils`
│   │   ├── utils       <- Helper functions and classes that implements Path Finding and Map manipulation
│   │   └── game.py     <- Highly abstract API for game generation out of maps
│   │
│   └── model_tomnet    <- Scripts to work with ToMnet-N model  [Train & Test how model predicts player in the game]
│       ├── layers      <- Custom layers for ToMnet-N construction
│       ├── tomnet.py   <- Create a ToMnet-N model
│       ├── train.py    <- Training algorithm for ToMnet-N
│       └── predict.py  <- API to utilise ToMnet-N
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

Documentation is generated with Sphinx and hosted on Read The Docs - *URL*

> `sphinx-build -b html docs/source/ docs/build/html` to generate web page