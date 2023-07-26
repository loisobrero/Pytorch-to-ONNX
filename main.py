import torch
import time
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic
from transformers import BertTokenizer, BertForSequenceClassification
from model import MovieReviewClassifier
from onnx_converter import convert_to_onnx

def synchronize_models():
    # Create an instance of the MovieReviewClassifier class
    review_classifier = MovieReviewClassifier()

    # Convert the PyTorch model to ONNX format
    convert_to_onnx(review_classifier.model)

    print("Models synchronized.")

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    # Quantize the ONNX model and save to disk
    quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)
    print("Quantized model saved.")

def run_pytorch_inference(review_text):
    # Load the pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Tokenize the input review text
    inputs = tokenizer(review_text, return_tensors='pt')

    # Run PyTorch inference
    outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_class_name = 'positive' if predicted_class == 1 else 'negative'

    return predicted_class_name

def run_onnx_inference(review_text, model_path):
    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(model_path)

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

    return predicted_class_name

def main():
    # Synchronize the PyTorch and ONNX models
    synchronize_models()

    # Quantize the ONNX model
    onnx_model_path = "movie_review_classifier.onnx"
    quantized_model_path = "movie_review_classifier_quantized.onnx"
    quantize_onnx_model(onnx_model_path, quantized_model_path)

    # Example review texts
    review_texts = [
        "This movie was amazing! I loved every minute of it.",
        "The acting was terrible, and the plot was boring.",
    ]

    for review_text in review_texts:
        # Perform PyTorch inference
        start_time = time.time()
        pytorch_prediction = run_pytorch_inference(review_text)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"PyTorch inference runtime: {runtime:.6f} seconds")
        print(f"PyTorch prediction: {pytorch_prediction}")

        # Perform ONNX inference
        start_time = time.time()
        prediction = run_onnx_inference(review_text, onnx_model_path)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"ONNX model predicted the review as: {prediction}")
        print(f"ONNX inference runtime: {runtime:.6f} seconds")

        # Perform quantized ONNX inference
        start_time = time.time()
        prediction = run_onnx_inference(review_text, quantized_model_path)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Quantized ONNX model predicted the review as: {prediction}")
        print(f"Quantized ONNX inference runtime: {runtime:.6f} seconds")

if __name__ == "__main__":
    main()
