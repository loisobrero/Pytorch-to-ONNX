from model import MovieReviewClassifier
from onnx_converter import convert_to_onnx

def main():
    review_classifier = MovieReviewClassifier()

    # Set the model to evaluation mode
    review_classifier.model.eval()

    # Example review texts
    review_texts = [
        "This movie was amazing! I loved every minute of it.",
        "The acting was terrible, and the plot was boring.",
    ]

    # Preprocess the data and run inference
    for review_text in review_texts:
        predicted_class = review_classifier.classify_review(review_text)

    # Convert the PyTorch model to ONNX format
    convert_to_onnx(review_classifier.model)

if __name__ == "__main__":
    main()