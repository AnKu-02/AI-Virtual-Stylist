import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIG ---
content_path = "stylemorph/outputs/01masked_full.jpg"
style_path = "stylemorph/styles/floral.jpg"
original_path = "stylemorph/test_images/1000089293.jpg"
mask_path = "stylemorph/outputs/01_mask.png"
output_path = "stylemorph/outputs/stylized_blend.jpg"
image_size = 512

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- IMAGE LOADING ---
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return loader(image).unsqueeze(0).to(device, torch.float)

def save_image(tensor, path):
    image = tensor.squeeze(0).cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy() * 255
    Image.fromarray(image.astype('uint8')).save(path)

# --- LOSS MODULES ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G.div(c * h * w)

# --- MODEL SETUP ---
def get_model_and_losses(cnn, style_img, content_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model[:i+1].to(device), style_losses, content_losses

# --- LOAD IMAGES ---
content_img = load_image(content_path)
style_img = load_image(style_path)
input_img = content_img.clone()

# --- LOAD MODEL ON GPU ---
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# --- STYLE TRANSFER ---
model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)

optimizer = optim.LBFGS([input_img.requires_grad_()])
num_steps = 200

print("[INFO] Stylizing masked person only...")
for step in range(num_steps):
    def closure():
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 100 + content_score
        loss.backward()
        return loss
    optimizer.step(closure)
    if step % 50 == 0:
        print(f"Step {step} complete")

# --- BLENDING WITH BACKGROUND ---
print("[INFO] Blending stylized person onto original background...")
stylized = input_img.clone().squeeze(0).cpu().clamp(0, 1)
original = transforms.ToTensor()(Image.open(original_path).convert("RGB").resize((stylized.shape[2], stylized.shape[1])))
mask = transforms.ToTensor()(Image.open(mask_path).convert("L").resize((stylized.shape[2], stylized.shape[1])))

mask = mask.expand_as(stylized)
output = stylized * mask + original * (1 - mask)
Image.fromarray((output.permute(1, 2, 0).detach().numpy() * 255).astype('uint8')).save(output_path)

print(f"[DONE] Final stylized image saved to: {output_path}")
