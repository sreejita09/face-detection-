# Face Mask Detection Project

This project aims to build a machine learning model that can analyze images of faces to determine whether the person is wearing a mask. The model will be trained on a dataset of images containing faces with and without masks.

## Project Structure

```
face-mask-detection
├── data
│   ├── raw                # Contains raw image data of faces
│   └── processed          # Contains processed image data for training
├── notebooks
│   └── exploration.ipynb  # Jupyter notebook for exploratory data analysis
├── src
│   ├── data_preprocessing.py  # Functions for loading and preprocessing data
│   ├── model.py               # Defines the machine learning model architecture
│   ├── train.py               # Responsible for training the model
│   ├── evaluate.py            # Functions to evaluate model performance
│   └── utils.py               # Utility functions used across the project
├── requirements.txt           # Lists Python dependencies for the project
├── README.md                  # Documentation for the project
└── .gitignore                 # Specifies files and directories to ignore by Git
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-mask-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place the raw images in the `data/raw` directory.
   - Run the data preprocessing script to process the images and save them in the `data/processed` directory.

## Usage

- Use the `notebooks/exploration.ipynb` for exploratory data analysis to gain insights into the dataset.
- Modify the `src/model.py` to define your desired model architecture.
- Train the model by running the `src/train.py` script.
- Evaluate the model's performance using the `src/evaluate.py` script.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## OUTPUT 

The plot above shows the training and validation accuracy over epochs.  
A steady increase in training accuracy and high validation accuracy indicate that the model is learning to distinguish between masked and unmasked faces effectively.  
Slight differences between the two curves are normal and suggest the model is generalizing well to unseen data.

<img width="1045" height="893" alt="image" src="https://github.com/user-attachments/assets/596bb1ae-69be-4567-8170-15ad9476136c" />
