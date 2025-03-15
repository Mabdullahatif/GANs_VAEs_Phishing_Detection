# GANs_VAEs_Phishing_Detection
Implementations of three distinct machine learning tasks GANs, VAEs, and Phishing Detection
1. **Generative Adversarial Networks (GANs)** for image generation.
2. **Variational Autoencoders (VAEs)** for both image generation and latent space analysis.
3. **Phishing Detection using VAEs**, which applies anomaly detection techniques to identify phishing URLs.

Additionally, the repository includes a detailed documentation manual with explanations and related questions.

## Project Structure
```
.
├── data_phishing.csv          # Phishing dataset used for training the VAE model
├── AI_GenAI_Manual.pdf        # Manual containing questions and explanations
├── 21L_6225_A1_GenAI.ipynb    # Jupyter Notebook containing the implementation
├── 21L_6225_A1_GenAI.pdf      # Detailed assignment documentation
```

## Tasks Covered
### 1. Generative Models
#### (a) Generative Adversarial Networks (GANs)
- Consists of a **Generator** and a **Discriminator**.
- The **Generator** learns to create realistic images from random noise.
- The **Discriminator** distinguishes between real and generated images.
- Used for generating images from **MNIST, FashionMNIST, and Digits datasets**.
- Evaluated using loss curves and image quality comparisons.

#### (b) Variational Autoencoders (VAEs)
- Uses an **encoder-decoder** architecture.
- The **Encoder** maps input data to a probabilistic latent space.
- The **Decoder** reconstructs inputs from the latent representation.
- Uses **PCA for latent space visualization**.
- Applied to **MNIST, FashionMNIST, and Digits datasets**.

### 2. Phishing Detection using Variational Autoencoders (VAEs)
- Uses an **anomaly detection approach**.
- The VAE is trained on **legitimate URLs**, learning to reconstruct them.
- **Phishing URLs are detected** based on high reconstruction errors.
- Preprocessing includes **feature extraction and standardization**.
- Achieved **70% accuracy** based on reconstruction error thresholds.

## Methodology
### 1. Data Preprocessing
- **Image Datasets (MNIST, FashionMNIST, Digits)**
  - Normalized pixel values for stable training.
  - Used PCA to visualize latent space representation.
- **Phishing Dataset (data_phishing.csv)**
  - Extracted relevant features from URLs.
  - Applied `StandardScaler` for normalization.
  - Divided dataset into training and validation sets.

### 2. GAN Architecture
- **Generator:** Converts random noise into realistic images.
- **Discriminator:** Determines real vs. fake images.
- **Loss Function:** Binary cross-entropy with adversarial training.
- **Evaluation:** Image quality and loss curve analysis.

### 3. VAE Architecture
- **Encoder:** Compresses input data into a latent space representation.
- **Decoder:** Reconstructs the original input from latent variables.
- **Loss Function:** Reconstruction loss + KL divergence.
- **Evaluation:** Reconstruction quality, latent space visualization.

### 4. Phishing Detection Model
- **VAE-based anomaly detection.**
- Trained only on legitimate URLs to reconstruct normal patterns.
- **Anomalies (phishing URLs) identified based on high reconstruction errors.**
- **Evaluation metrics:** ROC curve, classification report.

## Results
### 1. Image Generation (GANs & VAEs)
- **GANs** produced **sharper images**.
- **VAEs** generated images with **more structured latent space representations**.
- **Latent space analysis showed** clear clustering patterns in VAEs.
- **Datasets used:** MNIST, FashionMNIST, Digits.

### 2. Phishing Detection
- **ROC curve analysis** demonstrated performance trade-offs.
- **Achieved 70% accuracy**, balancing false positives and false negatives.
- **Challenges:** Handling imbalanced datasets, improving feature extraction.

## Running the Code
### Requirements
- Python 3.8+
- Jupyter Notebook / Google Colab / VS Code
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### How to Use
1. **Open `21L_6225_A1_GenAI.ipynb` in Jupyter Notebook, Google Colab, or VS Code.**
2. **Run the notebook cells in sequence.**
3. **Visualize results for GANs, VAEs, and phishing detection.**

## Future Work
- Improve phishing detection by integrating **additional URL-based features** (e.g., SSL certificates, domain age).
- Explore **advanced generative models** such as Wasserstein GANs (WGANs) and β-VAEs.
- Enhance dataset diversity for more **robust anomaly detection models**.

## Connect with Me
<p align="center">
    <a href="https://www.linkedin.com/in/mabdullahatif/">
        <img height="50" src="https://cdn2.iconfinder.com/data/icons/social-icon-3/512/social_style_3_in-306.png"/>
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://www.facebook.com/abdullahatif362/">
        <img height="50" src="https://cdn0.iconfinder.com/data/icons/social-flat-rounded-rects/512/facebook-64.png"/>
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://www.instagram.com/abdullah._.atif/">
        <img height="50" src="https://cdn2.iconfinder.com/data/icons/social-media-applications/64/social_media_applications_3-instagram-64.png"/>
    </a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://www.twitter.com/abd_allah_atif/">
        <img height="50" src="https://cdn3.iconfinder.com/data/icons/2018-social-media-logotypes/1000/2018_social_media_popular_app_logo_twitter-64.png"/>
    </a>
</p>
