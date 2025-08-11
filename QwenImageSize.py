import torch
import comfy.model_management
from nodes import MAX_RESOLUTION

class QwenImageSize:
    """A ComfyUI node that generates empty latents with specific sizes for different AI models.
    
    Supports multiple models including Qwen-Image, SDXL, and Flux, with their recommended resolutions.
    Provides options for orientation control and custom size overrides.
    """
    
    def __init__(self):
        """Initialize the node with the appropriate device."""
        self.device = comfy.model_management.intermediate_device()

    MODEL_RESOLUTIONS: dict[str, list[str]] = {
        "Qwen-Image": [
            # Vertical orientations (h > w)
            "928x1664 (9:16 Vertical)",
            "1056x1584 (2:3 Vertical)",
            "1140x1472 (3:4 Vertical)",
            # Square
            "1328x1328 (1:1 Square)",
            # Horizontal orientations (w > h)
            "1664x928 (16:9 Wide)",
            "1584x1056 (3:2 Wide)",
            "1472x1140 (4:3 Wide)",
        ],
        "SDXL": [
            # Vertical orientations (h > w)
            "704x1408 (1:2 Vertical)",
            "704x1344 (11:21 Vertical)",
            "768x1344 (4:7 Vertical)",
            "768x1280 (3:5 Vertical)",
            "832x1216 (13:19 Vertical)",
            "832x1152 (13:18 Vertical)",
            "896x1152 (7:9 Vertical)",
            "896x1088 (14:17 Vertical)",
            "960x1088 (15:17 Vertical)",
            "960x1024 (15:16 Vertical)",
            # Square
            "1024x1024 (1:1 Square)",
            # Horizontal orientations (w > h)
            "1024x960 (16:15 Wide)",
            "1088x960 (17:15 Wide)",
            "1088x896 (17:14 Wide)",
            "1152x896 (18:13 Wide)",
            "1152x832 (18:13 Wide)",
            "1216x832 (19:13 Wide)",
            "1280x768 (5:3 Wide)",
            "1280x704 (7:4 Wide)",
            "1344x768 (21:11 Wide)",
            "1344x704 (21:11 Wide)",
            "1408x704 (2:1 Wide)",
            "1408x640 (11:5 Wide)",
            "1472x704 (2:1 Wide)",
            "1536x640 (12:5 Wide)",
            "1600x640 (5:2 Wide)",
            "1664x576 (26:9 Wide)",
            "1728x576 (3:1 Wide)",
        ],
        "Flux": [
            # Vertical orientations (h > w)
            "768x1024 (3:4 Vertical)",
            "960x1280 (3:4 Vertical)",
            "960x1440 (2:3 Vertical)",
            "1024x1536 (2:3 Vertical)",
            # Square
            "512x512 (1:1 Square)",
            "768x768 (1:1 Square)",
            "1024x1024 (1:1 Square)",
            "1536x1536 (1:1 Square)",
            # Horizontal orientations (w > h)
            "1024x768 (4:3 Wide)",
            "1280x960 (4:3 Wide)",
            "1440x960 (3:2 Wide)",
            "1536x1024 (3:2 Wide)",
        ],
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (list(s.MODEL_RESOLUTIONS.keys()), {
                    "default": "Qwen-Image",
                }),
                # Model specific resolutions
                "qwen_resolution": (s.MODEL_RESOLUTIONS["Qwen-Image"], {
                    "default": s.MODEL_RESOLUTIONS["Qwen-Image"][0],
                }),
                "sdxl_resolution": (s.MODEL_RESOLUTIONS["SDXL"], {
                    "default": s.MODEL_RESOLUTIONS["SDXL"][0],
                }),
                "flux_resolution": (s.MODEL_RESOLUTIONS["Flux"], {
                    "default": s.MODEL_RESOLUTIONS["Flux"][0],
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                
                "width_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_RESOLUTION,
                    "step": 8,
                    "display": "number"
                }),
                "height_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_RESOLUTION,
                    "step": 8,  
                    "display": "number"
                }),
            }}

    RETURN_TYPES = ("LATENT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "latent/resolution"

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")

    def execute(self, model_type: str, qwen_resolution: str, sdxl_resolution: str, flux_resolution: str,
             batch_size: int, width_override: int = 0, height_override: int = 0) -> tuple:
        
        batch_size = max(1, batch_size)
        
        resolution_map = {
            "Qwen-Image": qwen_resolution,
            "SDXL": sdxl_resolution,
            "Flux": flux_resolution
        }
        resolution = resolution_map[model_type]

        try:
            width_str, height_str = resolution.split(" ")[0].split("x")
            width = int(width_str)
            height = int(height_str)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid resolution format: {resolution}. Expected format: 'WIDTHxHEIGHT (RATIO)'")

        if width_override > 0:
            width = width_override
        if height_override > 0:
            height = height_override
        
        if model_type != "Qwen-Image" and not (width_override > 0 or height_override > 0):
            width = (width // 64) * 64
            height = (height // 64) * 64

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples": latent}, width, height,) 

MISC_CLASS_MAPPINGS = {
    "QwenImageSize": QwenImageSize,
}

MISC_NAME_MAPPINGS = {
    "QwenImageSize": "📐 Qwen Image Size Picker",
}
