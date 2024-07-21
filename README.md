# Structure Informed Neural Networks

This is code repository and demo for the paper "Structure informed neural networks for boundary observation problems in fluid mechanics" which can be used to recreate the results in the paper. A demo for the first numerical example is available in the NumericalExample1Notebook.ipynb and also online at [demo link](https://drive.google.com/file/d/1r6fiRHBhgSGEU_26-fn7nWzZGopArEu0/view?usp=sharing). The second numerical example demo can be accessed in the notebook NumericalExample2Notebook.ipynb. The dataset and an example model will be automaticlly downloaded by running the second code cell, the size of the downloaded files will be <600 MB.

## Abstract of Structure informed neural networks for boundary observation problems in fluid mechanics

We address the problem of inferring the behaviour of a dynamical system within a domain using measurements taken only at its boundary. This is crucial in fluid mechanics, when non-intrusive sensors at the boundary are used to determine the state of the fluid inside its domain. We introduce a new data-driven method called Structure Informed Neural Networks (SINNs) whose key novelty is to model the transfer of information from  boundary-to-interior by embedding Elliptic Systems of PDEs into a classical encoder-decoder structure of a neural network. This approach ensures a well-posed network structure regardless of whether the data is well-posed or ill-posed. We show that SINNs are data-efficient and require no prior knowledge of the underlying PDEs. We demonstrate their effectiveness with two challenging examples: a nonlinear heat equation and the Navier-Stokes equations for flow past a body with variable geometry.

![SINNs abstract](https://github.com/aeroimperial-optimization/Structure-Informed-Neural-Networks/blob/main/mapping_schematic.jpg)

## Setup Instructions

### Prerequisites

- Ensure you have Python 3.11.7 installed on your system. You can download it from the [Python website](https://www.python.org/downloads/release/python-3117/).
- `git` should be installed on your system. You can download it from the [Git website](https://github.com/aeroimperial-optimization/Structure-Informed-Neural-Networks.git).

### Cloning the Repository

To clone the repository, open your terminal (or Command Prompt on Windows) and run the following command:

```bash
git clone https://github.com/username/repository.git
cd repository
```

### Setting Up the Virtual Environment

1.	Create a virtual environment using Python 3.11.7:

For Unix or MacOS:

```bash
python3.11 -m venv venv
```

For Windows:

```bash
py -3.11 -m venv venv
```

2.	Activate the virtual environment:
For Unix or MacOS:

```bash
source venv/bin/activate
```

For Windows:
```bash
venv\Scripts\activate
```

### Installing the Required Libraries

With the virtual environment activated, install the necessary libraries from requirements_base.txt:

For Windows:
```bash
pip install -r requirements_base.txt
```
