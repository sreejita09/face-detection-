# Face Mask Detection

This project is a deep learning model that detects whether a person in a photo is wearing a mask or not. It uses TensorFlow and Keras with a custom dataset of face images, organized into `with_mask` and `without_mask` folders.

## Features

- Loads and preprocesses images from folders
- Uses data augmentation for better generalization
- Convolutional Neural Network (CNN) for binary classification
- Plots training and validation accuracy

## Dataset Structure

Place your images in the following structure:

```
data/
  with_mask/
    img1.jpg
    img2.jpg
    ...
  without_mask/
    imgA.jpg
    imgB.jpg
    ...
```

## How to Run

1. **Install requirements**  
   Make sure you have Python 3 and install dependencies:
   ```
   pip install tensorflow matplotlib
   ```

2. **Prepare your data**  
   Extract your dataset so it matches the folder structure above.

3. **Train the model**  
   Run the script:
   ```
   python sample1_fd
   ```

4. **View results**  
   The script will plot training and validation accuracy.  
   *(You can add an output photo here after uploading it to GitHub.)*

## Model Architecture

- Data augmentation (random flip, rotation, zoom)
- 3 convolutional layers with max pooling
- Dense layer with dropout
- Output layer with sigmoid activation

## Example Output

*(Add your output photo here after uploading it to GitHub)*

## License

This project is for educational purposes.