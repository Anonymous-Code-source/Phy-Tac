# Phy-Tac

A deep learning framework for physics-informed tactile data generation using Variational Autoencoders (VAE) and Latent Diffusion Models (LDM).

## 📋 Overview
Phy-Tac is a research project that leverages generative models to process and synthesize tactile sensor data. The framework combines VAE for feature extraction and LDM for high-quality tactile data generation.


## 🏗️ Project Structure

```
Phy-Tac/
├── src/                          # Source code modules
├── dataset/                      # Dataset configurations and loaders
├── Assets/                       # Images, figures, and documentation assets
├── train_vae_xsense_SD_new_loss.py   # VAE training script with new loss function
├── train_ldm_xsense_new_vesion.py    # LDM training script (new version)
├── infer_vae_scores.py              # VAE inference and evaluation script
├── vae.yaml                         # VAE model configuration
├── ldm.yaml                         # LDM model configuration
├── index.html                       # Project webpage/documentation
└── README.md                        # This file
```

## 🚀 Features

- **Variational Autoencoder (VAE)**: Compress and encode tactile sensor data into latent representations
- **Latent Diffusion Model (LDM)**: Generate high-quality tactile data from latent space
- **Physics-Informed**: Incorporates physical constraints in the learning process
- **Configurable**: YAML-based configuration for easy experimentation
- **xsense Support**: Optimized for xsense tactile sensor data

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/lyushipeng-pixel/Phy-Tac.git
cd Phy-Tac

# Install dependencies (recommended: create a virtual environment first)
pip install -r requirements.txt  # Note: You may need to create this file
```

## 💻 Usage

### 1. Train VAE Model

Train the Variational Autoencoder with the new loss function:

```bash
python train_vae.py --config vae.yaml


Configuration options can be modified in `vae.yaml`:
- Learning rate
- Batch size
- Loss weights
- Network architecture
- Training epochs

### 2. Train LDM Model

Train the Latent Diffusion Model (new version):

```bash
python train_ldm.py --config ldm.yaml


Configuration options can be modified in `ldm.yaml`:
- Diffusion timesteps
- Noise schedule
- Conditioning parameters
- Model dimensions

### 3. Inference and Evaluation

Evaluate the trained VAE model and compute reconstruction scores:

```bash
python infer_vae_scores.py --model_path /path/to/checkpoint.pth
```

## 📊 Model Architecture

### VAE Architecture
- **Encoder**: Compresses tactile images into latent space
- **Decoder**: Reconstructs tactile data from latent representations
- **Loss Function**: Combines reconstruction loss, KL divergence, and physical constraints

### LDM Architecture
- **Diffusion Process**: Progressive denoising in latent space
- **Conditioning**: Can be conditioned on physical parameters or task descriptions
- **Sampling**: Efficient generation of high-quality tactile data

## 📁 Dataset Format

The project expects tactile sensor data in the following format:
- Images from xsense tactile sensors
- Organized by contact scenarios
- Metadata including force, position, and object properties

Please organize your data according to the specifications in `dataset/` directory.

## 🔧 Configuration Files

### vae.yaml
Main configuration for VAE training:
- Model hyperparameters
- Loss function weights
- Training parameters
- Data preprocessing settings

### ldm.yaml
Main configuration for LDM training:
- Diffusion model settings
- Noise scheduling parameters
- Conditioning strategies
- Sampling configurations

## 📈 Results

The framework achieves:
- High-quality tactile data reconstruction
- Realistic tactile data generation
- Physics-consistent outputs
- Efficient latent space representation



## 🔗 Related Projects

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Inspiration for latent diffusion approach
- [VAE variants](https://github.com/AntixK/PyTorch-VAE) - VAE implementations
- Tactile sensing datasets and benchmarks


## 🙏 Acknowledgments

This project builds upon advances in:
- Generative modeling (VAE, Diffusion Models)
- Tactile sensing and robotics
- Physics-informed machine learning

---

**Note**: This is a research project. For production use, additional testing and validation are recommended.

