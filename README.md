# Spam Message Detection Using NLP

## 1. Project Title
Spam Message Detection Using Natural Language Processing

## 2. Student Details
- Name: YOUR NAME
- Roll No.: YOUR ROLL NO.
- Branch/Semester: V Semester ETC
- Course: Natural Language Processing (ET5M004)

## 3. Problem Statement / Objective
Spam messages can contain unwanted advertisements, fake offers, fraudulent links, or misleading information. The objective of this project is to build a simple NLP-based system that classifies a text message as **SPAM** or **NOT SPAM**.

## 4. Introduction
Natural Language Processing (NLP) helps computers process and understand human language. In this project, text messages are converted into numerical features using TF-IDF and then classified using Logistic Regression.

## 5. NLP Technique / Method
The project uses:
- Text normalization through the TF-IDF vectorizer
- Stop-word removal
- Unigram and bigram features
- TF-IDF numerical representation
- Logistic Regression for classification

## 6. Dataset / Source
A small labelled educational dataset is included in `dataset/spam_messages.csv`.
- `ham` = normal / not spam message
- `spam` = unwanted or promotional message

The dataset is included for learning and demonstration. It is intentionally small so that the project is easy to understand.

## 7. Software / Tools / Libraries
- Python 3
- pandas
- scikit-learn
- Git
- GitHub
- VS Code / Jupyter Notebook (optional)

## 8. Methodology / Workflow
1. Load the labelled message dataset.
2. Separate messages (`X`) and labels (`y`).
3. Split the dataset into training and testing data.
4. Convert messages into TF-IDF vectors.
5. Train a Logistic Regression classifier.
6. Test the model on unseen messages.
7. Calculate accuracy and classification report.
8. Predict new messages as SPAM or NOT SPAM.

## 9. Project Structure
```text
NLP_Spam_Message_Detection/
├── README.md
├── dataset/
│   └── spam_messages.csv
├── source_code/
│   ├── spam_detector.py
│   └── app.py
├── notebooks/
├── output/
├── screenshots/
├── requirements.txt
└── .gitignore
```

## 10. Steps to Execute

### Step 1: Install Python
Install Python 3 on your computer.

### Step 2: Open the project folder
Open the folder in VS Code or Command Prompt.

### Step 3: Install libraries
```bash
pip install -r requirements.txt
```

### Step 4: Run the main program
```bash
python source_code/spam_detector.py
```

### Step 5: Run the interactive detector
```bash
python source_code/app.py
```

## 11. Sample Input
```text
Congratulations! You have won a free gift. Claim now!
```

## 12. Sample Output
```text
Result: SPAM
```

Another example:

```text
Please send me the notes after class.
```

Output:
```text
Result: NOT SPAM
```

## 13. Results / Observations
The system learns patterns from labelled messages and predicts whether a new message belongs to the spam or normal-message class. The exact accuracy is generated when the program is executed and is saved in `output/model_results.txt`.

## 14. Conclusion
This project demonstrates how NLP and machine learning can be combined to classify text messages. TF-IDF converts text into numerical features, while Logistic Regression learns the difference between spam and normal messages. The project can be extended by using a larger real-world SMS dataset and more advanced models.

## 15. Future Scope
- Use a larger dataset.
- Add a web interface.
- Compare multiple machine-learning algorithms.
- Add multilingual spam detection.
- Connect the model to an email or messaging application.

## 16. References / Resources
- Python documentation
- pandas documentation
- scikit-learn documentation
- Natural Language Processing course material
