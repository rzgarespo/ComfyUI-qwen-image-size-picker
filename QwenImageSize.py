import torch
import comfy.model_management
from nodes import MAX_RESOLUTION

class QwenImageSize:
    """A ComfyUI node that generates empty latents with specific sizes for different AI models.
    
    Supports multiple models including Qwen-Image, SDXL, Flux, and Flux2, with their recommended resolutions.
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
        "Z-Image": [
            # Vertical orientations (h > w)
            "720x1280 (9:16 Portrait)",
            "900x1600 (9:16 Portrait)",
            "832x1248 (2:3 Portrait)",
            "1024x1536 (2:3 Portrait)",
            "864x1152 (3:4 Portrait)",
            "960x1280 (3:4 Portrait)",
            # Square
            "1024x1024 (1:1 Square)",
            "1280x1280 (1:1 Square)",
            "1536x1536 (1:1 Square)",
            # Horizontal orientations (w > h)
            "1280x720 (16:9 Landscape)",
            "1600x900 (16:9 Landscape)",
            "1248x832 (3:2 Landscape)",
            "1536x1024 (3:2 Landscape)",
            "1152x864 (4:3 Landscape)",
            "1280x960 (4:3 Landscape)",
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
        "Flux2": [
            # Vertical orientations (h > w)
            "1408x2816 (1:2 Vertical)",
            "1408x2688 (11:21 Vertical)",
            "1536x2688 (4:7 Vertical)",
            "1536x2560 (3:5 Vertical)",
            "1664x2432 (13:19 Vertical)",
            "1664x2304 (13:18 Vertical)",
            "1792x2304 (7:9 Vertical)",
            "1792x2176 (14:17 Vertical)",
            "1920x2176 (15:17 Vertical)",
            "1920x2048 (15:16 Vertical)",
            # Square
            "2048x2048 (1:1 Square)",
            # Horizontal orientations (w > h)
            "2048x1920 (16:15 Wide)",
            "2176x1920 (17:15 Wide)",
            "2176x1792 (17:14 Wide)",
            "2304x1792 (18:13 Wide)",
            "2304x1664 (18:13 Wide)",
            "2432x1664 (19:13 Wide)",
            "2560x1536 (5:3 Wide)",
            "2560x1408 (7:4 Wide)",
            "2688x1536 (21:11 Wide)",
            "2688x1408 (21:11 Wide)",
            "2816x1408 (2:1 Wide)",
            "2816x1280 (11:5 Wide)",
            "2944x1408 (2:1 Wide)",
            "3072x1280 (12:5 Wide)",
            "3200x1280 (5:2 Wide)",
            "3328x1152 (26:9 Wide)",
            "3456x1152 (3:1 Wide)",
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
                "zimage_resolution": (s.MODEL_RESOLUTIONS["Z-Image"], {
                    "default": s.MODEL_RESOLUTIONS["Z-Image"][0],
                }),
                "sdxl_resolution": (s.MODEL_RESOLUTIONS["SDXL"], {
                    "default": s.MODEL_RESOLUTIONS["SDXL"][0],
                }),
                "flux_resolution": (s.MODEL_RESOLUTIONS["Flux"], {
                    "default": s.MODEL_RESOLUTIONS["Flux"][0],
                }),
                "flux2_resolution": (s.MODEL_RESOLUTIONS["Flux2"], {
                    "default": s.MODEL_RESOLUTIONS["Flux2"][0],
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

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height", "batch_size",)
    FUNCTION = "execute"
    CATEGORY = "latent/resolution"

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")

    def execute(self, model_type: str, qwen_resolution: str, zimage_resolution: str, sdxl_resolution: str,
             flux_resolution: str, flux2_resolution: str, batch_size: int, width_override: int = 0,
             height_override: int = 0) -> tuple:

        batch_size = max(1, batch_size)

        resolution_map = {
            "Qwen-Image": qwen_resolution,
            "Z-Image": zimage_resolution,
            "SDXL": sdxl_resolution,
            "Flux": flux_resolution,
            "Flux2": flux2_resolution,
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

        if model_type not in ("Qwen-Image", "Z-Image") and not (width_override > 0 or height_override > 0):
            width = (width // 64) * 64
            height = (height // 64) * 64

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples": latent}, width, height, batch_size,) 

class QwenImageSizeSimple:
    """A simplified ComfyUI node that generates empty latents using aspect ratios.

    The aspect ratio determines the actual resolution based on the selected model.
    Each model has different optimal resolutions for the same aspect ratio.
    """

    def __init__(self):
        """Initialize the node with the appropriate device."""
        self.device = comfy.model_management.intermediate_device()

    # Same model resolutions as the original node
    MODEL_RESOLUTIONS: dict[str, list[str]] = QwenImageSize.MODEL_RESOLUTIONS

    # Common aspect ratios supported by most models
    ASPECT_RATIOS = [
        "1:1 (Square)",
        "16:9 (Landscape)",
        "9:16 (Portrait)",
        "4:3 (Landscape)",
        "3:4 (Portrait)",
        "3:2 (Landscape)",
        "2:3 (Portrait)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(cls.MODEL_RESOLUTIONS.keys()), {
                    "default": "Qwen-Image",
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": cls.ASPECT_RATIOS[0],
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

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height", "batch_size",)
    FUNCTION = "execute"
    CATEGORY = "latent/resolution"

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")

    def _extract_aspect_ratio(self, resolution_str: str) -> str:
        """Extract the aspect ratio from a resolution string like '928x1664 (9:16 Vertical)'."""
        try:
            # Extract the part between parentheses
            ratio_part = resolution_str.split("(")[1].split(")")[0]
            # Get just the ratio (e.g., "9:16" from "9:16 Vertical")
            ratio = ratio_part.split()[0]
            return ratio
        except (IndexError, AttributeError):
            return None

    def _find_resolution_by_aspect_ratio(self, model_type: str, aspect_ratio: str) -> str:
        """Find the best resolution for a given model and aspect ratio."""
        # Extract the ratio from the aspect_ratio string (e.g., "9:16" from "9:16 (Portrait)")
        target_ratio = aspect_ratio.split()[0]

        # Get all resolutions for this model
        resolutions = self.MODEL_RESOLUTIONS.get(model_type, [])
        if not resolutions:
            raise ValueError(f"Unknown model type: {model_type}")

        # First, try to find an exact match for the aspect ratio
        for resolution in resolutions:
            ratio = self._extract_aspect_ratio(resolution)
            if ratio == target_ratio:
                return resolution

        # If no exact match, find the closest one (same orientation)
        # Determine if we want portrait (h > w) or landscape (w > h)
        is_portrait = ":" in target_ratio and int(target_ratio.split(":")[0]) < int(target_ratio.split(":")[1])

        # Try to find any resolution with the same orientation
        for resolution in resolutions:
            try:
                w_str, h_str = resolution.split(" ")[0].split("x")
                w, h = int(w_str), int(h_str)
                if is_portrait and h > w:
                    return resolution
                elif not is_portrait and w > h:
                    return resolution
            except (ValueError, IndexError):
                continue

        # Fallback to first resolution (usually square or close to square)
        return resolutions[0]

    def execute(self, model_type: str, aspect_ratio: str, batch_size: int,
             width_override: int = 0, height_override: int = 0) -> tuple:

        batch_size = max(1, batch_size)

        # Find the appropriate resolution based on model and aspect ratio
        resolution = self._find_resolution_by_aspect_ratio(model_type, aspect_ratio)

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

        if model_type not in ("Qwen-Image", "Z-Image") and not (width_override > 0 or height_override > 0):
            width = (width // 64) * 64
            height = (height // 64) * 64

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples": latent}, width, height, batch_size,)


MISC_CLASS_MAPPINGS = {
    "QwenImageSize": QwenImageSize,
    "QwenImageSizeSimple": QwenImageSizeSimple,
}

MISC_NAME_MAPPINGS = {
    "QwenImageSize": "📐 Qwen Image Size Picker",
    "QwenImageSizeSimple": "📐 Qwen Image Size Picker (Simple)",
}
