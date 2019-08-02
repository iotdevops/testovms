# Testing openvino model server

This is a simple application for testing a basic openvino model server hosting a classification model based on the Food-101 dataset.

Example usage:

python3 jpeg_classification.py --grpc_address [ip address of model server] --grpc_port 9001 \
--input_name input --output_name final_result --model_name [name of model to invoke] \
--images_list input_images.txt --labels food101_labels.txt


