import os
import sys

import torch
import pandas as pd
import numpy as np
import json

import shared.utils as su


# NOTE: Set the right paths to the data directories
data_dir = {
    "cia-ssv2": "/scratch/shared/beegfs/piyush/datasets/SSv2/20bn-something-something-v2/{}.webm",
    "cia-epic": "/scratch/shared/beegfs/piyush/datasets/EPIC-Kitchens-100/cut_clips/{}.MP4",
    "cia-charades": "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips/{}.mp4",
    "neg-coco": "/scratch/shared/beegfs/piyush/datasets/COCO2017/val2017/{}.jpg",
    "neg-msrvtt": "/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all/{}.mp4",
    "covr-webvid": "/datasets/WebVid/videos/{}.mp4"
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/')
    parser.add_argument('--model_name', type=str, default='tarsier2_7b')
    parser.add_argument("--csv_path", type=str, default='./data/nuanced_retrieval_inputs-val.csv')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()


    # Load CSV
    csv_path = args.csv_path
    csv_name = os.path.basename(csv_path).split('.')[0]
    assert os.path.exists(csv_path), f"CSV file does not exist: {csv_path}"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Remove rows with null values
    df = df.dropna()
    print(f"Number of rows after removing null values: {len(df)}")
    print('-' * 100)


    # Load model
    from modeling_tara import TARA
    model = TARA.from_pretrained(
        args.model_path,
        device_map='auto',
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    
    
    # Compute embeddings
    save_dir = f"{args.model_path}/embs"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.model_name}_{csv_name}_embeddings.pt"
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        print(f"Embeddings already exist at {save_path}")
        
        # Load it and only keep the rows with ids that are not in the embeddings
        embeddings = torch.load(save_path)
        df = df[~df['id'].isin(list(embeddings.keys()))]
        print(f"Number of rows to compute: {len(df)}")
        
        if len(df) == 0:
            print("All embeddings already computed")
            exit()
    else:
        embeddings = {}
    
    if args.debug:
        # Debug on a particular example
        debug_id = "092151_092200/1015822471|change the setting to a nightclub"
        row = df[df['id'] == debug_id].iloc[0].to_dict()
        z = model.encode_vision_with_text(
            data_dir[row['source']].format(eval(row['value'])['video']),
            eval(row['value'])['text'],
        ).squeeze(0).cpu().float()
        import ipdb; ipdb.set_trace()
    
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing embeddings'):
        row = df.iloc[i].to_dict()
        
        # try:
            
        if row['modality'] in ["text", "text-standard", "text-negation"]:
            z = model.encode_text(row['value']).squeeze(0).cpu().float()
        elif row['modality'] == "image":
            image_path = data_dir[row['source']].format(row['value'])
            z = model.encode_image(image_path).squeeze(0).cpu().float()
        elif row['modality'] == "video":
            video_path = data_dir[row['source']].format(row['value'])
            z = model.encode_vision(video_path).squeeze(0).cpu().float()
        elif row['modality'] == "video-text":
            z = model.encode_vision_with_text(
                data_dir[row['source']].format(eval(row['value'])['video']),
                eval(row['value'])['text'],
            ).squeeze(0).cpu().float()
        else:
            raise ValueError(f"Invalid modality: {row['modality']}")

        z = torch.nn.functional.normalize(z, dim=-1)
        embeddings[row['id']] = z

        # except Exception as e:
        #     print(f"Error computing embedding for {row['id']}: {e}")
        #     continue
    
    # Save embeddings
    torch.save(embeddings, save_path)
    print(f"Saved embeddings to {save_path}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of rows: {len(df)}")
