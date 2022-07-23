## Emory Math REU 2022: Jacobian-Free Backpropagation (JFB) for Implicit Networks
Repository for Team JFB working on the project **Fast Training of Implicit Networks with Applications in Inverse Problems** during the Emory Computatational
Math REU.
The team is mentored by Dr. Samy Wu Fung and consists of 3 undergraduate students:
1. Linghai Liu, Brown University
2. Shuaicheng Tong, UCLA
3. Lisa Zhao, UC Berkeley

In this project, we use a recently proposed JFB approach to solve the computational challenges of image deblurring using implicit networks.

## Running the Code
Initialize virtual environment: 
``py -m venv env``
This gives you a virtual environment under the directory ``env``
To use virtual environment, ``source ./env/bin/activate``. 
To stop it, ``deactivate``. 

Use the following steps to install packages:
``pip install -r ../requirements.txt``

``pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
``

Note: whenever you run the code, please watch out for the file names where the model loaded/saved. 
Use the following to train the model

- pretrain dncnn: ``python ./scripts/script_pretrain.py``. The output is supposed to be in ``./results/dncnn_pretrain/``
- train DE-GRAD with JFB: ``python ./scripts/script_fixlr.py``. The output is supposed to be in ``./results/degrad_fixlr/``
