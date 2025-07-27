# 🧠 Handwritten Digit Classifier with CNN (MNIST)

This project is a simple yet powerful Convolutional Neural Network (CNN) built using TensorFlow and Keras to classify handwritten digits (0–9) from the MNIST dataset. It also includes functionality to predict digits from custom images.

---

#Features

- Trains a CNN model on the MNIST dataset
- Evaluates the model's accuracy on test data
- Predicts custom handwritten digits from external images
- Clean, simple structure for beginners and educational use

# Model Architecture

- Reshape input to `(28, 28, 1)`
- 3 Convolutional layers with ReLU
- 2 MaxPooling layers
- Dense layers with 64 and 10 units
- Uses `SparseCategoricalCrossentropy` and `Adam` optimizer


#⚠️ Python Compatibility

> 🚨 **Important:**  
This project is compatible with **Python 3.11**.  


## 📁 Project Structure

```
.
├── mnist_cnn.py         # Main Python script
├── 5.jpg              # (Optional) Your custom digit image
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```


### 1. Create and activate virtual environment
```bash
python -m venv cnn_mnist
cnn_mnist\Scripts\activate   # On Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python mnist_cnn.py
```


# Author

Developed by **Ali Abbas Abbbod**  
With ❤️ using Python 3.11 and TensorFlow

---

# License

This project is open-source under the [MIT License](LICENSE).