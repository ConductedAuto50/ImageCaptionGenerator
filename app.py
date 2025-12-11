import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
import os


# Global variables for model
model = None
processor = None
device = None


def load_model_once():
    """Load model once when the app starts."""
    global model, processor, device
    
    if model is None:
        print("Loading model...")
        model_path = "./model/blip_fine_tuned_model"
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")


def add_overlay_image(image, style):
    """Add overlay image to the base image based on the selected style."""
    overlay_image_paths = {
        "Funny": "funny.jpeg",
        "Formal": "formal.jpeg",
        "Poetic": "poetic.jpeg"
    }
    
    # Get overlay image path
    overlay_path = overlay_image_paths.get(style)
    if not overlay_path:
        return image
    
    # Create a copy of the image
    img_copy = image.copy()
    
    try:
        # Load the overlay image
        overlay = Image.open(overlay_path).convert("RGBA")
        
        # Resize overlay to fit the base image (e.g., 30% of image width)
        base_width = img_copy.size[0]
        overlay_width = int(base_width * 0.2)
        overlay_height = int(overlay.size[1] * (overlay_width / overlay.size[0]))
        overlay = overlay.resize((overlay_width, overlay_height), Image.Resampling.LANCZOS)
        
        # Convert base image to RGBA if needed
        if img_copy.mode != 'RGBA':
            img_copy = img_copy.convert('RGBA')
        
        # Position overlay at top centre with some padding
        position = ((base_width - overlay_width) // 2, 20)
        
        # Paste overlay with alpha transparency
        img_copy.paste(overlay, position, overlay)
        
        # Convert back to RGB
        img_copy = img_copy.convert('RGB')
        
    except FileNotFoundError:
        print(f"Warning: Overlay image not found at {overlay_path}. Using original image.")
        return image
    except Exception as e:
        print(f"Error adding overlay: {e}. Using original image.")
        return image
    
    return img_copy


def generate_caption_with_overlay(image, style, max_length, num_beams):
    """
    Generate caption for the image with optional overlay.
    
    Args:
        image: PIL Image
        style: Style of overlay ("None", "Funny", "Formal", "Poetic")
        max_length: Maximum caption length
        num_beams: Number of beams for generation
    
    Returns:
        caption: Generated caption string
    """
    # Load model if not already loaded
    load_model_once()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Add overlay if style is selected
    processed_image = image
    if style != "None":
        processed_image = add_overlay_image(image, style)
        
        # Save the image with overlay
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # Process image for model
    inputs = processor(images=processed_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate caption
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption


# Custom CSS for styling
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 2em;
}

.gradio-container {
    font-family: 'Arial', sans-serif;
}

#image-upload {
    border: 2px dashed #667eea;
    border-radius: 10px;
}

#caption-output {
    font-size: 1.3em;
    font-weight: 500;
    color: #333;
    padding: 20px;
    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    border-radius: 10px;
    border-left: 4px solid #667eea;
}

.generate-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
}
"""


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 id='title'>Social Media Caption Generator</h1>")
    gr.Markdown("<p id='subtitle'>Upload an image, choose a style, and let AI create the perfect caption!</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", 
                label="📸 Upload Your Image",
                elem_id="image-upload",
                sources=["upload", "webcam", "clipboard"]
            )
            
            style_input = gr.Radio(
                choices=["None", "Funny", "Formal", "Poetic"],
                value="None",
                label="🎨 Choose Style",
                info="Choose a tone for your caption"
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                max_length_input = gr.Slider(
                    minimum=20,
                    maximum=150,
                    value=50,
                    step=10,
                    label="Maximum Caption Length",
                    info="Longer captions provide more detail"
                )
                
                num_beams_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Generation Quality (Beams)",
                    info="Higher values = better quality but slower"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Caption",
                variant="primary",
                elem_classes="generate-btn"
            )
        
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="📝 Generated Caption",
                placeholder="Your caption will appear here...",
                lines=8,
                elem_id="caption-output"
            )
    

    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_caption_with_overlay,
        inputs=[image_input, style_input, max_length_input, num_beams_input],
        outputs=caption_output
    )


if __name__ == "__main__":
    print("Starting Social Caption Generator...")
    print("Loading model (this may take a moment)...")
    load_model_once()
    print("Model loaded! Launching interface...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, css=custom_css)
