import torch

def convert_to_onnx(model):
    # Example input sequence length (adjust this according to your requirements)
    max_sequence_length = 64

    # Create a random input tensor for export with the correct size
    inputs = torch.randint(0, 1000, (1, max_sequence_length)).to(torch.long)

    # Set the model to CPU before exporting to ONNX
    model.cpu()

    onnx_model_path = "movie_review_classifier.onnx"
    torch.onnx.export(model, inputs, onnx_model_path, export_params=True)
    print("The model has been successfully converted to ONNX format.")