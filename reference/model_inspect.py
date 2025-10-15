"""
Video Model Inspection Script
This script demonstrates how to inspect and analyze various video classification models
from the Transformers library, including TimeSformer, V-JEPA, ViViT, and VideoMAE.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoVideoProcessor,
    TimesformerForVideoClassification,
    VivitImageProcessor,
    VivitModel,
    VideoMAEForVideoClassification,
)


# ==============================================================================
# Section 1: TimeSformer Model Inspection
# ==============================================================================

def inspect_timesformer():
    """
    Inspect TimeSformer model architecture and outputs.
    
    Returns:
        dict: Model outputs including logits, hidden states, and attentions
    """
    print("\n" + "="*80)
    print("TimeSformer Model Inspection")
    print("="*80)
    
    # Generate random video data
    video = list((np.random.rand(8, 224, 224, 3) * 255).astype(np.uint8))
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
    
    # Enable attention and hidden state outputs
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    print(f"Output attentions: {model.config.output_attentions}")
    print(f"Output hidden states: {model.config.output_hidden_states}")
    
    # Process video and get predictions
    inputs = processor(images=video, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
    
    # Analyze outputs
    print(f"\nOutput keys: {outputs.keys()}")
    print(f"Attention shape (first layer): {outputs.attentions[0].shape}")  # (8, 12, 197, 197)
    print(f"Hidden states shape (first layer): {outputs.hidden_states[0].shape}")  # (1, 1569, 768)
    print(f"Number of hidden state layers: {len(outputs.hidden_states)}")  # 13 (last one is embedding)
    print(f"Number of attention layers: {len(outputs.attentions)}")  # 12
    
    return outputs


# ==============================================================================
# Section 2: V-JEPA 2 Model Inspection
# ==============================================================================

def inspect_vjepa2():
    """
    Inspect V-JEPA 2 model architecture and outputs.
    Requires torchcodec to be installed for video decoding.
    
    Returns:
        tuple: (encoder_outputs, predictor_outputs, similarity_score)
    """
    print("\n" + "="*80)
    print("V-JEPA 2 Model Inspection")
    print("="*80)
    
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        print("Warning: torchcodec not installed. Skipping V-JEPA 2 inspection.")
        return None
    
    # Load model and processor
    hf_repo = "facebook/vjepa2-vitl-fpc64-256"
    model = AutoModel.from_pretrained(hf_repo).to("cuda")
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    
    print(f"Model device: {model.device}")
    
    # Load and process video
    video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
    vr = VideoDecoder(video_url)
    frame_idx = np.arange(0, 64)  # Sample 64 frames
    video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
    video = processor(video, return_tensors="pt").to(model.device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**video)
    
    # V-JEPA 2 encoder outputs (same as calling `model.get_vision_features()`)
    encoder_outputs = outputs.last_hidden_state
    
    # V-JEPA 2 predictor outputs
    predictor_outputs = outputs.predictor_output.last_hidden_state
    
    print(f"Encoder outputs shape: {encoder_outputs.shape}")  # (1, 8192, 1024)
    print(f"Predictor outputs shape: {predictor_outputs.shape}")  # (1, 8192, 1024)
    
    # Calculate cosine similarity between encoder and predictor
    similarity = F.cosine_similarity(encoder_outputs, predictor_outputs, dim=2).mean()
    print(f"Cosine similarity (encoder vs predictor): {similarity.item():.4f}")
    
    return encoder_outputs, predictor_outputs, similarity


# ==============================================================================
# Section 3: ViViT Model Inspection
# ==============================================================================

def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (list[int]): List of frame indices to decode.
        
    Returns:
        np.ndarray: Array of decoded frames of shape (num_frames, height, width, 3).
    """
    import av
    
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    
    Args:
        clip_len (int): Total number of frames to sample.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Maximum allowed index of sample's last frame.
        
    Returns:
        list[int]: List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def inspect_vivit():
    """
    Inspect ViViT model architecture and outputs.
    Requires av (PyAV) to be installed for video decoding.
    
    Returns:
        torch.Tensor: Last hidden states from the model
    """
    print("\n" + "="*80)
    print("ViViT Model Inspection")
    print("="*80)
    
    try:
        import av
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Warning: PyAV or huggingface_hub not installed. Skipping ViViT inspection.")
        return None
    
    np.random.seed(0)
    
    # Download and open video clip (300 frames, 10 seconds at 30 FPS)
    file_path = hf_hub_download(
        repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    )
    container = av.open(file_path)
    
    # Sample 32 frames
    indices = sample_frame_indices(
        clip_len=32, 
        frame_sample_rate=1, 
        seg_len=container.streams.video[0].frames
    )
    video = read_video_pyav(container=container, indices=indices)
    
    # Load model and processor
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
    
    # Prepare video for the model
    inputs = image_processor(list(video), return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state
    print(f"Last hidden states shape: {list(last_hidden_states.shape)}")
    
    return last_hidden_states


# ==============================================================================
# Section 4: VideoMAE Model Inspection
# ==============================================================================

def inspect_videomae():
    """
    Inspect VideoMAE model architecture and outputs.
    
    Returns:
        dict: Model outputs including logits, hidden states, and attentions
    """
    print("\n" + "="*80)
    print("VideoMAE Model Inspection")
    print("="*80)
    
    # Generate random video data (16 frames, 224x224, RGB, float32 in [0,1])
    num_frames = 16
    H = W = 224
    video = [np.random.rand(H, W, 3).astype(np.float32) for _ in range(num_frames)]
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        attn_implementation='eager'
    )
    model.eval()
    
    # Enable attention and hidden state outputs
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    print(f"Output attentions: {model.config.output_attentions}")
    print(f"Output hidden states: {model.config.output_hidden_states}")
    
    # Process video and get predictions
    inputs = processor(video, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    pred = logits.argmax(-1).item()
    print(f"Predicted class: {model.config.id2label[pred]}")
    
    # Analyze outputs
    print(f"\nOutput keys: {outputs.keys()}")
    print(f"Hidden states shape (last layer): {outputs.hidden_states[-1].shape}")
    print(f"Number of hidden state layers: {len(outputs.hidden_states)}")
    print(f"Attention shape (first layer): {outputs.attentions[0].shape}")
    print(f"Number of attention layers: {len(outputs.attentions)}")
    
    return outputs


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("Video Model Inspection Script")
    print("Inspecting multiple video classification models from Transformers library")
    
    # Run all inspections
    try:
        inspect_timesformer()
    except Exception as e:
        print(f"Error inspecting TimeSformer: {e}")
    
    try:
        inspect_vjepa2()
    except Exception as e:
        print(f"Error inspecting V-JEPA 2: {e}")
    
    try:
        inspect_vivit()
    except Exception as e:
        print(f"Error inspecting ViViT: {e}")
    
    try:
        inspect_videomae()
    except Exception as e:
        print(f"Error inspecting VideoMAE: {e}")
    
    print("\n" + "="*80)
    print("Inspection Complete")
    print("="*80)



