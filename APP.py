from flask import Flask, render_template, jsonify
import speech_recognition as sr
from transformers import XLNetTokenizer, XLNetLMHeadModel
from langdetect import detect
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

model_name = "xlnet-large-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/record_and_summarize")
def record_and_summarize():
    # Function to record audio
    def record_audio():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Starting recording...")
            try:
                audio = recognizer.listen(source)
                print("Finish record")
                recorded_text = recognizer.recognize_google(audio)
                return recorded_text.strip()  # Strip leading/trailing whitespace
            except sr.UnknownValueError:
                print("Cannot recognize audio")
                return ""  # Return empty string on recognition failure
            except sr.RequestError as e:
                print("Failed request; {0}".format(e))
                return ""  # Return empty string on request error
            except Exception as e:
                print("Errors occur: {0}".format(e))
                return ""  # Return empty string on other errors

    # Function to perform text summarization
    def summarize_text(text):
        # Text processing and lemmatization
        doc = nlp(text)
        lemmatized_text = " ".join(
            [lemmatizer.lemmatize(token.text.lower()) for token in doc])

        # Detect language of the input text
        language = detect_language(text)
        if not language:
            print("Language detection failed. Using default language.")
            language = "en"  # Use default language

        # Tokenize input text
        input_ids = tokenizer.encode(
            lemmatized_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        try:
            summary_ids = model.generate(
                input_ids, max_length=150, min_length=50, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print("Error occurred during summarization:", e)
            return None

    # Record audio
    recorded_text = record_audio()

    # Check for empty recorded text
    if not recorded_text:
        return jsonify({"recorded_text": "", "summary": "Recorded text is empty. Please speak again."})

    print("Recorded Text:", recorded_text)  # Print the recorded text

    # Perform text summarization
    summarized_text = summarize_text(recorded_text)
    if summarized_text:
        return jsonify({"recorded_text": recorded_text, "summary": summarized_text})
    else:
        return jsonify({"recorded_text": recorded_text, "summary": "Failed to generate summary."})


def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print("Language detection failed:", e)
        return None


if __name__ == "__main__":
    app.run(debug=True)
