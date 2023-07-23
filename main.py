import torch
import time
import onnxruntime
from transformers import BertTokenizer, BertForSequenceClassification
from model import MovieReviewClassifier
from onnx_converter import convert_to_onnx

def synchronize_models():
    # Create an instance of the MovieReviewClassifier class
    review_classifier = MovieReviewClassifier()

    # Convert the PyTorch model to ONNX format
    convert_to_onnx(review_classifier.model)

    print("Models synchronized.")

def run_pytorch_inference(review_text):
    # Load the pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Tokenize the input review text
    inputs = tokenizer(review_text, return_tensors='pt')

    # Run PyTorch inference and measure the runtime
    outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_class_name = 'positive' if predicted_class == 1 else 'negative'

    print(f"The PyTorch model predicted the review as: {predicted_class_name}")
    return predicted_class_name

def run_onnx_inference(review_text):
    # Load the ONNX model
    onnx_model_path = "movie_review_classifier.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Print the input names and shapes of the ONNX model
    print("Input names and shapes of the ONNX model:")
    for input_info in ort_session.get_inputs():
        print(f"{input_info.name}: {input_info.shape}")

    # Tokenize the input review text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(review_text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

    # Run inference with ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: inputs['input_ids'].numpy()}
    if len(ort_session.get_inputs()) == 2:
        ort_inputs[ort_session.get_inputs()[1].name] = inputs['attention_mask'].numpy()

    ort_outputs = ort_session.run(None, ort_inputs)

    # Get the predicted class
    logits = torch.tensor(ort_outputs[0])
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_class_name = 'positive' if predicted_class == 1 else 'negative'

    print(f"The ONNX model predicted the review as: {predicted_class_name}")
    return predicted_class_name

def main():
    # Synchronize the PyTorch and ONNX models
    synchronize_models()

    # Example review texts
    review_texts = [
        "This movie was amazing! I loved every minute of it.",
        "The acting was terrible, and the plot was boring.",
    ]

    for review_text in review_texts:
        # Perform PyTorch inference
        start_time = time.time()
        run_pytorch_inference(review_text)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"PyTorch inference runtime: {runtime:.6f} seconds")

        # Perform ONNX inference
        start_time = time.time()
        run_onnx_inference(review_text)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"ONNX inference runtime: {runtime:.6f} seconds")

if __name__ == "__main__":
    main()
