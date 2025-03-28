# Fake News Detector

This is a web application built with Streamlit that classifies news articles as "Fake" or "Real" using a pre-trained machine learning model. The app takes text input from users and leverages natural language processing and machine learning to provide predictions.

## Setup

Follow these steps to clone the repository and set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: The application uses NLTK for text preprocessing. The required NLTK data (stopwords and wordnet) will be downloaded automatically the first time you run the app if they are not already present on your system.*

## Running the Application

Once the setup is complete, start the Streamlit app with the following command:

```bash
streamlit run app.py
```

After running the command, open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the application.

## Usage

1. Enter the text of a news article into the provided text area.
2. Click the "Classify" button to see whether the article is classified as "Fake" or "Real."

## Project Structure

Here’s an overview of the key files in the project:

- **`app.py`**: The main Streamlit application file that runs the web app.
- **`news_classifier_model.pkl`**: The pre-trained machine learning model used for classification.
- **`requirements.txt`**: A list of Python dependencies required to run the project.
- **`README.md`**: This file, containing project information and instructions.

## Contributing

If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

### Notes for Implementation

- **Customize the Repository URL**: Replace `yourusername` in the `git clone` command with your actual GitHub username or organization name.
- **Ensure Dependencies**: Before sharing your project, create a `requirements.txt` file with all necessary libraries (e.g., `streamlit`, `nltk`, `joblib`, etc.). You can generate this file by running `pip freeze > requirements.txt` in your project environment.
- **License File**: If you choose to include the MIT License, add a `LICENSE` file to your repository with the standard MIT License text.

This `README.md` provides a clear and concise guide for users to understand, clone, and run your Fake News Detection project locally.