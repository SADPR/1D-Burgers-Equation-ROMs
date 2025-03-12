from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image

def visualize_autoencoder():
    # Create a new directed graph
    dot = Digraph(comment='Convolutional Autoencoder')

    # Input layer
    dot.node('Input', 'Input Layer\n1x512', shape='box')

    # Encoder
    dot.node('Conv1', 'Conv1d\n16x512\nKernel:3, Padding:1', shape='box')
    dot.node('MaxPool1', 'MaxPool1d\n16x256\nStride:2', shape='box')
    dot.node('Conv2', 'Conv1d\n32x256\nKernel:3, Padding:1', shape='box')
    dot.node('MaxPool2', 'MaxPool1d\n32x128\nStride:2', shape='box')
    dot.node('Conv3', 'Conv1d\n64x128\nKernel:3, Padding:1', shape='box')
    dot.node('MaxPool3', 'MaxPool1d\n64x64\nStride:2', shape='box')

    # Fully connected layers
    dot.node('FC1', 'Fully Connected\n64x64\n→ Latent Dim', shape='ellipse')
    dot.node('FC2', 'Fully Connected\nLatent Dim\n→ 64x64', shape='ellipse')

    # Decoder
    dot.node('DeConv1', 'ConvTranspose1d\n64x128\nKernel:4, Stride:2, Padding:1', shape='box')
    dot.node('DeConv2', 'ConvTranspose1d\n32x256\nKernel:4, Stride:2, Padding:1', shape='box')
    dot.node('DeConv3', 'ConvTranspose1d\n16x512\nKernel:4, Stride:2, Padding:1', shape='box')
    dot.node('Output', 'Output Layer\n1x512', shape='box')

    # Corrected connection of nodes using edges as pairs
    dot.edge('Input', 'Conv1')
    dot.edge('Conv1', 'MaxPool1')
    dot.edge('MaxPool1', 'Conv2')
    dot.edge('Conv2', 'MaxPool2')
    dot.edge('MaxPool2', 'Conv3')
    dot.edge('Conv3', 'MaxPool3')
    dot.edge('MaxPool3', 'FC1')
    dot.edge('FC1', 'FC2')
    dot.edge('FC2', 'DeConv1')
    dot.edge('DeConv1', 'DeConv2')
    dot.edge('DeConv2', 'DeConv3')
    dot.edge('DeConv3', 'Output')

    # Save and render the graph to a file
    dot.render('autoencoder_architecture', format='png', cleanup=False)

    # Display the generated graph using Matplotlib
    img = Image.open('autoencoder_architecture.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Visualize the architecture
visualize_autoencoder()

