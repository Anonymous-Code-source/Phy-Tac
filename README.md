# Phy-Tac

A deep learning framework for physics-informed tactile data generation using Variational Autoencoders (VAE) and Latent Diffusion Models (LDM).

## 📋 Overview

Phy-Tac is a research project that leverages generative models to process and synthesize tactile sensor data. The framework combines VAE for feature extraction and LDM for high-quality tactile data generation, with specific focus on xsense tactile sensor data.

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

### Recommended Dependencies

```bash
pip install torch torchvision
pip install numpy pandas
pip install opencv-python
pip install matplotlib seaborn
pip install pyyaml
pip install tqdm
```

## 💻 Usage

### 1. Train VAE Model

Train the Variational Autoencoder with the new loss function:

```bash
python train_vae_xsense_SD_new_loss.py --config vae.yaml
```

Configuration options can be modified in `vae.yaml`:
- Learning rate
- Batch size
- Loss weights
- Network architecture
- Training epochs

### 2. Train LDM Model

Train the Latent Diffusion Model (new version):

```bash
python train_ldm_xsense_new_vesion.py --config ldm.yaml
```

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{phy-tac2024,
  author = {Lyu, Shipeng},
  title = {Phy-Tac: Physics-Informed Tactile Data Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/lyushipeng-pixel/Phy-Tac}
}
```

## 📄 License

Please check the repository for license information.

## 👤 Author

**Lyu Shipeng** ([@lyushipeng-pixel](https://github.com/lyushipeng-pixel))

## 🔗 Related Projects

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Inspiration for latent diffusion approach
- [VAE variants](https://github.com/AntixK/PyTorch-VAE) - VAE implementations
- Tactile sensing datasets and benchmarks

## 📮 Contact

For questions or collaborations, please:
- Open an issue on GitHub
- Contact the author through GitHub profile

## 🙏 Acknowledgments

This project builds upon advances in:
- Generative modeling (VAE, Diffusion Models)
- Tactile sensing and robotics
- Physics-informed machine learning

---

**Note**: This is a research project. For production use, additional testing and validation are recommended.

