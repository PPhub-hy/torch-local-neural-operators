# Implementation of  the local neural operator

This page releases codes and data of an academic paper, 'Local neural operator for solving transient partial differential equations on varied domains'.
With the provided codes, one can quickly reproduce the training and validation of local neural operator.
Also, two demos for applying the well-trained LNOs to respectively solve the 2-D flows around a square and the NACA0012 airfoils are provided.

## Environment

Pytorch 1.6.0

## Data

We provide full testing data for learning the Navier-Stokes equations (mu = 0.002), and a single sample for training (including the velocity fields of 8s).

To reproduce the reported results, please download the provided data from

URL: https://pan.baidu.com/s/1QMH_1VvlODgivHOY-aTj1g

code: j3h7

Unzip the data files at ```/Train_Validation/Data/*```

## Training and Validation

The Python script main.py does both the network training and validation.
Please change the function inside the script

Train and validate a new network named 'newLNO'
```
python main.py -n newLNO
```

Validated the trained network provided in models/
'''
python main.py -n NS_200ep_LNO-legendre2_n16m8k2
'''

## Applications

Apply the trained LNO to solve the 2-D flow around a square

```
python mainSquareCylinder.py
```

Apply the trained LNO to solve the 2-D flow across a cascade airfoil

```
python mainCascade.py
```

## GIF Demos

Validation: flow in 2-D square domain with periodical boundaries (mu=0.002)

 ![image]( demos/Re500Case3_u_L.gif)
 
 ![image]( demos/Re500Case3_v_L.gif)
 
Application 1: 2-D flow around a square

 ![image]( demos/SquareCylinder_L.gif)

Application 2: 2-D flow across a cascade airfoil

 ![image]( demos/Airfoil_L.gif)

## Paper and citation 

This paper is now published as a preprint on arXiv.org.
Full text can be downloaded at https://doi.org/10.48550/arXiv.2203.08145.

Please cite this paper if any part of the codes here is used in your research. Great thanks to you!
