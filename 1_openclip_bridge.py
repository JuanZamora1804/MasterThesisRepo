import torch
from PIL import Image
import open_clip
import pathlib as Path
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from prompts import *

scales = ["micro", "meso", "macro", "overall"]
scales_prompts = [
    micro_prompts,
    meso_prompts,
    macro_prompts,
    overall_prompts,
]

positions = ["below", "close", "above"]
positions_prompts = [
    below_prompts,
    close_prompts,
    above_prompts,
]


def load_image_names(bridge_name):
    """Load image filenames from Excel sheet."""
    df = pd.read_csv(
        rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_image_list.csv"
    )
    df = df.dropna(subset=["Filename"])
    image_names = df["Filename"].astype(str).tolist()

    return image_names


def ensable_classifier(model, tokenizer, prompts_list, device):
    weights = []

    for prompts in prompts_list:
        texts = tokenizer(prompts).to(device)

        with torch.no_grad(), torch.autocast("cuda"):
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # Average the embeddings (Prompt Ensemble)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

        weights.append(class_embedding)

    return torch.stack(weights)  # Stack to get shape [n_classes, embed_dim]


def run_openclip(bridge_name, model_number, scales, scales_prompts, positions, positions_prompts):
    # os.system("cls" if os.name == "nt" else "clear")

    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    openclip_models = pd.read_csv(r"C:\Users\juanj\Desktop\openclip_bridgesfm\openclip_available_models.csv")
    # openclip_models = pd.read_csv(r"/home/jjzj/CODES/OpenCLIP/openclip_available_models.csv")
    
    model_name = openclip_models.iloc[model_number]["name"]
    pretrained_name = openclip_models.iloc[model_number]["pretrained"]

    print(f"\nProcessing Bridge: {bridge_name}")
    print(f"Using model: {model_name} - pretrained on: {pretrained_name}")

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_name, device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        # Create classifier weights with prompt ensembling
        scale_weights = ensable_classifier(model, tokenizer, scales_prompts, device)
        position_weights = ensable_classifier(
            model, tokenizer, positions_prompts, device
        )

        # Load image
        image_names = load_image_names(bridge_name)

        # print(f"Scale Weights Shape: {scale_weights.shape}")
        # print(f"Position Weights Shape: {position_weights.shape}")

        results = []

        safe_model_name = model_name.replace("/", "-")  # Clean model names.
        file_name = rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\{bridge_name}_classification_results_with_CLIP_{safe_model_name}_{pretrained_name}.csv"

        for n in tqdm(range(len(image_names)), desc="Processing images"):
            image_path = Path.Path(
                rf"C:\Users\juanj\Desktop\Bridges\{bridge_name}\Input images\{image_names[n]}" 
                # f"C:\\Users\\juanj\\Desktop\\bridgesfm\\data\\0_bridge_{bridge_name}\\img_input\\{image_names[n]}"
                # f"/home/jjzj/CODES/BridgeSfm/0_bridge_{bridge_name}/img_input/{image_names[n]}"
            )
            image = (
                preprocess(Image.open(image_path).convert("RGB"))
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad(), torch.autocast("cuda"):
                image_embedding = model.encode_image(image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                # Compute similarity with scale and position weights
                scale_logits = 100.0 * image_embedding @ scale_weights.T
                position_logits = 100.0 * image_embedding @ position_weights.T

                scale_probs = scale_logits.softmax(dim=-1)
                position_probs = position_logits.softmax(dim=-1)

            best_scale_idx = torch.argmax(scale_probs, dim=-1).item()
            best_scale = scales[best_scale_idx]

            best_position_idx = torch.argmax(position_probs, dim=-1).item()
            best_position = positions[best_position_idx]

            file_exists = os.path.isfile(file_name)

            results = pd.DataFrame(
                {
                    "index": [n],
                    "filename": [image_names[n]],
                    "scale": [best_scale],
                    "scale_probabilities": np.max(scale_probs.cpu().numpy().flatten()),
                    "position": [best_position],
                    "position_probabilities": np.max(position_probs.cpu().numpy().flatten()),
                }
            )

            results.to_csv(file_name, mode="a", header=not file_exists, index=False)

            # Free image tensor memory inside the loop
            del image

        print(f"Results saved to {file_name}")

        # Clean up to free GPU memory
        del model, preprocess, tokenizer, scale_weights, position_weights
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {e}, with model #{model_number}: {model_name}\n")
        pass


# modelos_openclip = [0] #, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 120]

# for bridge_name in ["Gaskugel", "Nibelungen"]:
    # for i in modelos_openclip:

# bridge_name = "1_Avignon Viaducts"
folders = []
for entry in os.listdir(r"C:\Users\juanj\Desktop\Bridges"):
    if os.path.isdir(os.path.join(r"C:\Users\juanj\Desktop\Bridges", entry)):
        folders.append(entry)

for folder in folders:


# os.makedirs(f"./results3/{bridge_name}", exist_ok=True)
# print(f"\nProcessing Bridge: {bridge_name}, Model Number: {i+1}/{len(modelos_openclip)}\n")
    run_openclip(
        bridge_name=folder,
        model_number=57,
        scales=scales,
        scales_prompts=scales_prompts,
        positions=positions,
        positions_prompts=positions_prompts,
    )
