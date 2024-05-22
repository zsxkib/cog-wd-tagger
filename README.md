# WD Image Tagger 🏷️🖼️

> [![Replicate - WD Image Tagger](https://replicate.com/zsxkib/wd-image-tagger/badge)](https://replicate.com/zsxkib/wd-image-tagger)

The **WD Image Tagger** is a powerful AI model that automatically analyzes and tags your images with descriptive labels. It's trained on a large dataset of anime-style images and can recognize a wide range of content, including general attributes, characters, and age ratings. 

This tool was developed using resources and models available on [SmilingWolf's wd-tagger Hugging Face Space](https://huggingface.co/spaces/SmilingWolf/wd-tagger), ensuring state-of-the-art performance and ease of use.


Whether you're managing a large image library, looking to generate accurate prompts for an AI art model, or want to quickly filter out potentially sensitive content, the WD Image Tagger can help streamline your workflow.

## Features

- 🌟 Pre-trained on a diverse dataset of anime images
- 🏷️ Tags images with general attributes, characters, and content ratings
- 🔍 Supports multiple state-of-the-art model architectures like SwinV2, ConvNext, and ViT
- ⚙️ Adjustable tag probability thresholds for fine-grained control over results
- 🧮 Optional MCut algorithm for automatic threshold optimization
- 🗂️ Filter tags by category to focus on what's most relevant to you
- 🔌 Easy integration into existing applications via a simple API

## Getting Started

To start tagging your images with the WD Image Tagger:

1. Upload your image
2. Select the pre-trained model you'd like to use
3. Adjust the tag probability thresholds and category filters as needed
4. Let the model analyze your image and output the relevant tags

The model will return a list of tags, each with a confidence score and category label (general, character, or rating).

## Pre-trained Models

The WD Image Tagger comes with several pre-trained model options, each with its own strengths:

- `SwinV2`: A powerful and accurate model architecture well-suited for most use cases
- `ConvNext`: An efficient model that offers a good balance of speed and accuracy
- `ViT` (Vision Transformer): A transformer-based model that excels at capturing global context

Models are provided in both the latest Dataset v3 series and the earlier v2 series. The v3 models were trained on a larger and more diverse dataset, while the v2 models offer compatibility with older workflows.

## Acknowledgments

The WD Image Tagger was trained using the [SW-CV-ModelZoo](https://github.com/SmilingWolf/SW-CV-ModelZoo) toolkit, with TPUs generously provided by the TRC program. Special thanks to the researchers and engineers who made this powerful tool possible!

## Learn More

For more technical details on the available models and their expected performance, check out the [WD Image Tagger GitHub repository](https://github.com/SmilingWolf/wd-v1-4-vit-tagger-v2).

---

We hope the WD Image Tagger helps make your image analysis workflows faster and more effective. If you have any questions or feedback, don't hesitate to reach out!