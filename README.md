# Dynamics-Learning Nonlinear System Identification

This repository contains a hands-on implementation of nonlinear system identification using simulated data from a cart–pole system. It compares a neural network, Gaussian Processes, and a physics-informed residual model.

## What is included

- A continuous-time, frictionless cart–pole model.
- Fourth-order Runge–Kutta integration to generate discrete-time trajectories.
- Separate training, validation, and test trajectories.
- One-step identification with an MLP, four independent exact GPs, and a physics-plus-residual model.
- One-step RMSE and NRMSE, GP predictive uncertainty, and open-loop rollout evaluation.
- Optional experiments for measurement-noise robustness and data efficiency.
- Multi-step fine-tuning for the MLP and hybrid models.

## Models

The state is defined as

$$
x_k = [p_k,\;\theta_k,\;v_k,\;\omega_k]^{\mathsf T},
$$

and the applied cart force is $u_k$. Each model receives

$$
z_k = [x_k^{\mathsf T},\;u_k]^{\mathsf T}
$$

and predicts the state increment

$$
\Delta x_k = x_{k+1}-x_k.
$$

The next state is reconstructed as $\widehat{x}_{k+1}=x_k+\widehat{\Delta x}_k$.

The notebook includes three models:

1. **MLP:** a feed-forward neural network trained with mean squared error.
2. **Gaussian Process:** one exact GP for each state-increment component, using a Matérn kernel with ARD.
3. **Hybrid model:** an RK4 prediction from a slightly mismatched physical model, corrected by a neural network that learns the residual.

The hybrid model is given a deliberately imperfect physics prior. Its cart mass is increased by 8%, pole mass is reduced by 8%, and centre-of-mass distance is increased by 5% relative to the simulator. The residual network is trained to compensate for this difference. This is an example of residual learning; it is not a physical-parameter estimation method.

## Data and evaluation

The default experiment uses a sampling period of **0.05 s** and trajectories of **3 s**. Data are collected near the upright position using state feedback and a bounded, piecewise-constant disturbance. The default split contains 20 training trajectories, 5 validation trajectories, and 8 test trajectories. The groups are generated independently with different random seeds.

Scalers are fitted on training data only. Validation data are used for early stopping and model selection. The test trajectories are held out until final evaluation.

The notebook reports one-step RMSE and NRMSE for each state. It also reports GP uncertainty using predictive variance, negative log-likelihood, and 95% interval coverage. Open-loop rollouts use the same initial state and recorded input sequence for each model. This separates one-step prediction accuracy from the ability to reproduce a trajectory over time.

The known-physics reference is included as a sanity check. Since it uses the same equations and parameters as the simulator, its error should be close to numerical precision. It is not a learned model and is not a competitor in an unknown-system identification problem.

## Run the notebook

The notebook is designed to run in Google Colab or a local Jupyter environment.

1. Open `dynamics-learning.ipynb` in Colab or Jupyter.
2. Run the first cell to install the required packages.
3. Restart the runtime if the environment asks you to do so.
4. Run the cells from top to bottom.

The main experiment trains all three models and evaluates their one-step and multi-step predictions. The measurement-noise and data-efficiency experiments are disabled by default because they retrain the models and can take longer, especially for exact GPs. To run them, change the corresponding flag in the notebook to `True`.

## Requirements

The notebook uses:

- Python 3
- NumPy
- Matplotlib
- scikit-learn
- PyTorch
- GPyTorch

For a local environment, install the packages with:

```bash
pip install -r requirements.txt
```

## Repository files

```text
.
├── README.md
├── requirements.txt
└── dynamics-learning.ipynb
```

The English notebook is intended for the main project page. The Persian notebook contains the same implementation with Persian explanations.

## Scope and limitations

This is a simulated benchmark, not an experiment with a physical cart–pole. The default data cover a limited region around the upright position. The plant model is frictionless and does not include actuator dynamics, sensor bias, or unmodelled delays. The optional measurement-noise experiment uses independent Gaussian noise and assumes that the applied force is known.

Exact GPs can become slow as the number of training samples grows. The notebook uses separate scalar-output GPs for the four state increments, so computation increases with both the data size and the number of outputs.

The results should be interpreted within these assumptions. Low one-step error does not necessarily mean that a model will give an accurate long-horizon rollout.

## Acknowledgement and licensing

This is an educational implementation developed from a cart–pole system-identification exercise. If any code, figures, or explanations are adapted from the course material or its teaching repository, add the original source link and follow its attribution and licence terms before publishing.
