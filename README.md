# Fashion Classification App

## Overview

This project is a Streamlit-based web application that uses a deep learning model to classify clothing images into different categories. The app allows users to upload images of clothing or footwear and receive predictions about the category of the item.

![image](https://github.com/user-attachments/assets/48388c7b-d309-41d1-b9a5-0a7f6d2ee24f)


## Features

- Image upload and classification
- Real-time predictions
- Visualization of confidence levels for each category
- User-friendly interface with instructions and tips

## Categories

The model classifies images into the following categories:
- Boys-Apparel
- Boys-Footwear
- Girls-Apparel
- Girls-Footwear

## Technologies Used

- Python
- Streamlit
- TensorFlow
- NumPy
- Pillow (PIL)
- Matplotlib

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/Adekunle-Habeeb/Fashion-classification-app
   cd clothing-classification-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Ensure you have the trained model file (`1.keras`) in the correct location:
   ```
   saved model/1.keras
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the file uploader to select an image of clothing or footwear.

4. Click the "Classify Image" button to see the prediction results.

## Tips for Best Results

- Use clear, well-lit images
- Ensure the item of clothing or footwear is the main focus of the image
- Avoid cluttered backgrounds
- Try different angles if the initial classification seems incorrect

## Contributing

Contributions to improve the app are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [Streamlit](https://www.streamlit.io/) for making it easy to create web apps with Python
- The deep learning model was trained using [TensorFlow](https://www.tensorflow.org/)

## Contact

Your Name - adekunle22taiwo@gmail.com

Project Link: [(https://github.com/Adekunle-Habeeb/Fashion-classification-app)](https://github.com/Adekunle-Habeeb/Fashion-classification-app))
