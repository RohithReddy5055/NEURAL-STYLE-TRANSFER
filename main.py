import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import argparse
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
torch.set_default_device(device)

def load_image(image_name, size=512):
    """Load and preprocess image."""
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    """Display an image tensor."""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class ContentLoss(nn.Module):
    """Calculates the content loss between two feature maps."""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    """Computes the Gram Matrix of a feature map."""
    a, b, c, d = input.size() # a=batch size, b=number of feature maps, (c,d)=dimensions of a feature map
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """Calculates the style loss using Gram Matrices."""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    """Normalization module for VGG pre-trained weights."""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# VGG19 default content and style layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """Constructs the model with content and style loss layers."""
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last loss layer
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    """Creates the LBFGS optimizer for the input image."""
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Main optimization loop for style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run[0]}:")
                print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                print()

            return style_score + content_score

        optimizer.step(closure)

    # Final clamp
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def main():
    parser = argparse.ArgumentParser(description='Neural Style Transfer using PyTorch.')
    parser.add_argument('--content', type=str, default='content.jpg', help='Path to content image')
    parser.add_argument('--style', type=str, default='style.jpg', help='Path to style image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--steps', type=int, default=300, help='Number of optimization steps')
    parser.add_argument('--size', type=int, default=512, help='Image size (resized to square)')
    parser.add_argument('--style_weight', type=float, default=1000000, help='Weight for style loss')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    
    args = parser.parse_args()

    if not os.path.exists(args.content) or not os.path.exists(args.style):
        print("Content or Style image not found. Please provide valid paths.")
        print("Example: python main.py --content my_photo.jpg --style my_style.jpg")
        return

    # Load pre-trained VGG19
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    # Load images
    content_img = load_image(args.content, size=args.size)
    style_img = load_image(args.style, size=args.size)
    
    # Initialize input image (clone of content image)
    input_img = content_img.clone()

    # Run Style Transfer
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, 
                                num_steps=args.steps,
                                style_weight=args.style_weight,
                                content_weight=args.content_weight)

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    imshow(content_img, title='Content Image')
    plt.subplot(1, 3, 2)
    imshow(style_img, title='Style Image')
    plt.subplot(1, 3, 3)
    imshow(output, title='Output Image')
    
    # Save output
    output_image = output.cpu().clone().squeeze(0)
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(args.output)
    print(f"Saved stylized image as {args.output}")
    plt.show()

if __name__ == '__main__':
    main()
