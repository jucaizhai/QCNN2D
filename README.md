# QCNN2D

QCNN2D is a quantum convolutional neural network implementation designed for solving partial differential equations (PDEs). This project leverages the power of quantum computing to improve the efficiency and accuracy in solving complex PDEs compared to traditional methods.

## Requirements
- **Python Version:** 3.8 or later
- **Dependencies:**
  - numpy
  - scipy
  - pytorch
  - pennylane

### Important Files
Ensure that the following files are present in the project directory:
- **AI4PDEs_bounds.py**
- **AI4PDEs_utils.py**

## Setting Up the Environment
1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
2. **Activate the Virtual Environment**:

On Windows:
bash
venv\Scripts\activate
On macOS and Linux:
bash
source venv/bin/activate
Install Dependencies:

bash
pip install -r requirements.txt
Running the Project
To run the project, use the following command:

bash
python QCNN2D.py
Public Entry Points
This project provides two main public entry points:

compare_scipy_and_qft_consistency(): Compare the results from SciPy methods with Quantum Fourier Transform (QFT) methods for consistency.
run_flow_past_block_demo(): Execute a demonstration of flow past a block within the quantum framework.
Supported Backends
The QCNN2D project supports the following backends:

scipy: Standard backend utilizing SciPy for computations.
pytorch: Leverages PyTorch for efficient tensor operations.
quantum: Implements quantum algorithms for problem-solving.
quantum_lcu: Utilizes the Linear Combination of Unitaries approach for quantum computations.
Note: PennyLane is required for all quantum backends. Make sure to install it as part of your dependencies.

For further information, please refer to the documentation or source code.

Code
Please confirm you want Copilot to make this change in the jucaizhai/QCNN2D repository on branch main.
