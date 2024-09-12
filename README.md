# Custom-LLM
This repository implements a language model primarily using the decoder side of the transformer architecture. The model is trained to generate text by predicting the next character in a sequence, utilizing a simple text encoding technique. The encoding converts characters into numerical indices, which can then be processed by the model. By mapping each unqiue character to an integer, the model is able to learn the relationships between different characters and generate coherent text sequences.

## Table of Content
1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Training](#training)
4. [Inference](#inference)

## Repository Structure

```
Custom-LLM/
├── checkpoints/
│   ├── .gitkeep
│   ├── best_model.pt
│   ├── last_model.pt
├── data/
│   ├── downloaded_text.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### Step-by-Step Guide to Set Up `nvidia-driver-560`, `CUDA 11.8`, and `cuDNN 8.7` on Ubuntu

**1. Uninstall Existing Nvidia Drivers (if any)** <br>
Before installing new drivers, ensure that any existing Nvidia drivers are removed.
```
sudo apt-get --purge remove "*nvidia*"
sudo apt-get autoremove
sudo apt-get autoclean
```
<br>Reboot the system<br> 
```
sudo reboot
```

**2. Add the Nvidia Package Repository**<br>
Add the Nvidia repository to get the lastest drivers.
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```

**3. Install Nvidia Driver 560**
<br>Install the `nvidia-driver-560` package.
```
sudo apt-get install nvidia-driver-560
```
<br>Reboot the system<br>
```
sudo reboot
```

**4. Verify Nvidia Driver Installation**
<br>After rebooting, verify that driver is installed correctly.
```
nvidia-smi
```
This should display the details of your GPU and the installed driver.

**5. Install CUDA 11.8 toolkit**
<br>Download the CUDA 11.8 toolkit from the official Nvidia website.
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```
Make the installer executable and run it
```
chmod +x cuda_11.8.0_520.61.05_linux.run
sudo ./cuda_11.8.0_520.61.05_linux.run
```
Since you already installed `nvidia-driver-560`, you can choose not to install the driver when prompted.

**6. Set Up Environment Variables**
<br>Add CUDA paths to your `.bashrc` file for persistent access.
```
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**7. Verify CUDA Installation**
<br>Check if the CUDA toolkit is correctly installed.
```
nvcc --version
```
This should show CUDA version as `11.8`.

**8. Install cuDNN 8.7**
<br>Download cuDNN 8.7 from Nvidia website (requires login) and extract it.
```
tar -xzvf cudnn-linux-x86_64-8.7.*.tgz
```
Copy the cuDNN files to the CUDA directory.
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

**9. Verify cuDNN Installation**
<br>To verify, check cuDNN version using
```
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### Install Required Python Packages
To run the code, you need the following dependencies:
- Python 3.9+
- CUDA 11.8
- cuDNN 8.7

**1. Create a virtual environment**
Create a virtual environment to avoid dependency conflict with existing packages in the base environment.
```
conda create -n custom_llm python=3.9
```
**2. Setup the repository**
```
git clone https://github.com/adityarajsahu/Custom-LLM.git
cd Custom-LLM/
```
**3. Install dependencies**
<br> Run the following command to install all the necessary packages.
```
pip install -r requirements.txt
```

## Training

To train the language model, run the `src/train.py` script. The script will automatically save the best model and the last model during training.

```
python src/train.py
```
The hyperparameters such as `batch_size`, `learning_rate`, `n_head`, and others are configurable via the `config.py` file.

If you want to train the language model on your own text file, add the text file to `data/` and change the file path in `config.py` accordingly.

The script automatically saves:
- The **best model** (based on the validation loss) to `checkpoints/best_model.pt`.
- The **last model** after training is completed to `checkpoints/last_model.pt`.

## Inference
To generate text using the trained model, run the `inference.py` script. You can specify a prompt to generate text based on the trained model.
```
python src/inference.py
```

### Example:
```
prompt = "Hello! Can you see me?"
```
You can customize the prompt inside the `inference.py` file and generate additional characters using the trained model. 