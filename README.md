# Facial Emotion Recognition (FER) Pipeline

This project implements a complete facial emotion recognition pipeline using OpenCV for real-time face detection and DeepFace for emotion classification. It features two distinct modes: a live interactive webcam feed and a batch processing tool for evaluating image datasets.

## Prerequisites & Installation

Before running the scripts, ensure you have Python installed and the necessary libraries downloaded. 

Run the following command in your terminal to install all required dependencies:

```bash
pip install opencv-python deepface matplotlib pandas scikit-learn seaborn
```

## Project Structure

To successfully run the Batch Mode, your project directory must be structured like this:

```bash
your_project_folder/
|
|-- main.py                  # The main entry point script
|-- partner_a_vision.py      # Face detection and webcam loop logic
|-- partner_b_ai.py          # DeepFace classification and batch analysis logic
|
|-- dataset/                 # REQUIRED FOR BATCH MODE
    |-- angry/               # Put test images inside their respective emotion folders
    |-- happy/               
    |-- neutral/             
    |-- ...
```

## How to Run the Program

The application is controlled via the ```main.py``` script using the ```--mode``` argument.

**1. Live Webcam Mode**

This mode opens your webcam, detects faces in real-time using Haar Cascades, and displays the dominant emotion along with a live confidence bar chart for all 7 universal emotions.

```bash
python main.py --mode webcam
```

Controls:

  Press ```'s'``` to save a screenshot of the current frame to your project folder.
  Press ```'q'``` to safely quit the webcam feed and close the window.

**2. Batch Analysis Mode**

This mode iterates through the dataset/ folder, analyzes every image, and evaluates the DeepFace model's accuracy against the true emotion labels (folder names). 

```bash
python main.py --mode batch
```

**Outputs:**

Once the batch analysis finishes running, it will automatically generate and save two files to your project folder:

  _results.csv:_ A spreadsheet detailing the true label vs. predicted label for every individual image.
  
  _confusion_matrix.png:_ A Seaborn heatmap visually representing the model's classification accuracy and common errors.


## Key Features

**A1: Preprocessing:** Images are resized to 224x224, converted to grayscale, and enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization) before classification.

**A2: Face Detection:** Utilizes OpenCV's Haar Cascade, tuned to an optimal scaleFactor=1.1, minNeighbors=5, and minSize=(30, 30) to balance sensitivity and reduce false positives.

**B1/B2: AI Classification & UI:** DeepFace integration evaluates 7 universal emotions, cached to update once per second to prevent CPU throttling while maintaining a smooth webcam framerate.

**B3: Batch Evaluation:** Automatically handles edge cases (like tiny 48x48 pixel dataset images) by passing the full image to the classifier if the face detector fails to find a bounding box.
