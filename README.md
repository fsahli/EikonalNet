# EikonalNet

Code for the paper: Sahli Costabal F, Yang Y, Perdikaris P, Hurtado DE and Kuhl E (2020) [Physics-Informed Neural Networks for Cardiac Activation Mapping.](https://www.frontiersin.org/articles/10.3389/fphy.2020.00042/abstract) Front. Phys. 8:42. doi: 10.3389/fphy.2020.00042

Dependencies:

- tensorflow v1.0
- pyDOE: `pip install pydoe`
- matplotlib, numpy, scipy
- Non-parametric Entropy Estimation Toolbox: https://github.com/gregversteeg/NPEET

There are two files containing the models: `models_tf.py`, which is a single neural network and `models_para_tf.py`, which implements the randomized prior functions and trains multiple neural networks in parallel.

Within the `models_tf.py` there is a 2D and a 3D model:
- `Eikonal2DnetCV2`, which has the inputs:
  - `x, y`: location of the collocation points to enforce the residual penalty. Each of them must be of shape `N_colloc, 1`.
  - `x_e, y_e`: location of the data points. Each of them must be of shape `N_data, 1`.
  - `T_e`: activation times data. Must be of shape `N_data, 1`.
  - `layers`: list with the number of neurons of each layer for the activation times, for example `[2,20,20,1]`.
  - `CVlayers`: list with the number of neurons of each layer for the conduction velocity, for example `[2,5,5,1]`.
  - `C`: maximum conduction velocity.
  - `alpha`: regularization coefficient for the conduction velocity.
  - `alphaL2`: regularization coefficient for the weights of the neural network.
  - `jobs`: number of cpus to use to train the model.
  
 - `Eikonal3DnetCV2`, which has the inputs:
    - `X`: location of the collocation points to enforce the residual penalty. For a triangular mesh, this corresponds to the centroid of each triangle. Must be of shape `N_colloc, 3`.
    - `normals`: unit vectors representing the normal of the surface at the points `X`. For a triangular mesh, this corresponds to the normal of each triangle. Must be of shape `N_colloc, 3`
    - `X_e`: location of the data points. Each of them must be of shape `N_data, 3`.
    - `T_e`: activation times data. Must be of shape `N_data, 1`.
    - `layers`: list with the number of neurons of each layer for the activation times, for example `[3,20,20,1]`.
    - `CVlayers`: list with the number of neurons of each layer for the conduction velocity, for example `[3,5,5,1]`.
    - `C`: maximum conduction velocity.
    - `alpha`: regularization coefficient for the conduction velocity.
    - `alphaL2`: regularization coefficient for the weights of the neural network.
    - `jobs`: number of cpus to use to train the model.
  
Within the `models_para_tf.py` there is a 2D and a 3D model as well:

- `Eikonal2DnetCV2RPF`, which has the additional inputs than `Eikonal2DnetCV2`:
  - `Batch`: the number of neural networks to train in parallel.
  - `noise_level`: which is the expected standard deviation of the measurements. 
  
- `Eikonalnet3DRPF`, which has the additional inputs than `Eikonal3DnetCV2`:
  - `Batch`: the number of neural networks to train in parallel.
  - `noise_level`: which is the expected standard deviation of the measurements. 
  
The files `2Dexample.ipynb` demonstrates the use of a single neural network and the file `active_learning_2Dexample.py` demontrates the use of multiple neural networks with active learning.
  
