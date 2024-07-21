# Structure Informed Neural Networks

This is code repository and demo for the paper "Structure informed neural networks for boundary observation problems in fluid mechanics" which can be used to recreate the results in the paper. A demo for the first numerical example is available in the NumericalExample1Notebook.ipynb and also online at [demo link](https://drive.google.com/file/d/1r6fiRHBhgSGEU_26-fn7nWzZGopArEu0/view?usp=sharing). The code for generating the results in the second numerical example is given in the notebook NumericalExample1Notebook.ipynb but the dataset is not shared on Github due to size limits. The dataset in a format which matches the code can be requested from the author at jakub.horsky.cz@gmail.com.

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