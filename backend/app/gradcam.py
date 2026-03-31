import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import io
from PIL import Image


class GradCAM:
    """
    gradmap implementation for DenseNet-121
    Hooks into the last denseblock to extract activation maps.

    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # target: last dense block in DenseNet-121
        target_layer = self.model.densenet.features.denseblock4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        generate grad-cam heatmap for a given class index

        Args:
            tensor: preprocessed input tensor (1, 3, 224, 224)
            class_idx: index of the target pathology class (0–13)

        Returns:
            heatmap as numpy array (224, 224), values in [0, 1]
        """
        self.model.model.zero_grad()
        output = self.model.model(tensor)

        # backprop on the target class score
        score = output[0, class_idx]
        score.backward()

        # global average pooling over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


def overlay_heatmap(heatmap: np.ndarray, original_image: Image.Image) -> Image.Image:
    """
    overlay gradcam heatmap on the original image using a jet colormap
    """
    orig_np = np.array(original_image.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlaid)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(model, tensor: torch.Tensor, original_image: Image.Image) -> str:
    """
    generate gradcam for the top predicted class
    overlay on the original image return as base64 PNG string
    """
    cam_generator = GradCAM(model)
    heatmap = cam_generator.generate(tensor, class_idx=0)
    overlaid = overlay_heatmap(heatmap, original_image)
    return image_to_base64(overlaid)