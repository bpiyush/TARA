import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Dict, List
from termcolor import colored
import random


import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaConfig, 
    LlamaForCausalLM,
)
from torchvision.transforms.v2 import (
    ToPILImage,
)
import decord
from decord import VideoReader

decord.bridge.set_bridge("torch")

# TODO: need to use these directly
from tarsier.modeling_tarsier import TarsierForConditionalGeneration
from tarsier.processor import Processor
# from utils.model import transform_pixel_values


EOL_PROMPTS = {
    'text': '<sent>\nSummary above sentence in one word:',
    'image': '<image>\nSummary above image in one word:',
    'video': '<video>\nSummary above video in one word:',
}


def transform_pixel_values(pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    # NOTE: this function doesn't accept unbatched inputs
    # pixel_values should be uint8 of (B, T, C, H, W)
    if isinstance(pixel_values, list):
        pixel_values = torch.stack(pixel_values)

    if pixel_values.ndim == 4:
        # pixel_values is (B, C, H, W)
        # (B, C, H, W) -> (B, 1, C, H, W)
        pixel_values = pixel_values.unsqueeze(1)
    elif pixel_values.ndim == 5:
        # pixel_values is (B, T, C, H, W)
        pass
    else:
        raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
    return pixel_values


base_registry = {}
class BaseModel(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            base_registry[cls.ARCHITECTURE] = cls
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        **kwargs):
        print(colored(f'[ MODEL ] Loading {cls.__name__} from {model_name_or_path} [..............]', 'yellow'))

        return cls(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)


class BaseModelForTARA(BaseModel):
    
    ARCHITECTURE = "TarsierForConditionalGeneration"
    LLM_CLASS = LlamaForCausalLM
    MLLM_CLASS = TarsierForConditionalGeneration

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["text"]} ASSISTANT: '
        return prompt
    
    @property
    def image_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["image"]} ASSISTANT: '
        return prompt
    
    @property
    def video_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["video"]} ASSISTANT: '
        return prompt

    @property
    def video_edit_eol_prompt(self):
        prompt = "Source video: <video>\nEdit instruction: <sent>\n"\
        "Look at the attached video carefully. The provided text is instruction to edit the video. "\
        "Imagine this edit instruction being applied to the provided video frame.\n"\
        "Summarize the resulting edited video in one word:"
        prompt = f"USER: {prompt} ASSISTANT: "
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: Optional[bool] = None,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):

        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'
            model_config = None
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        else:
            model_config = LlavaConfig.from_pretrained(
                model_name_or_path,
                # trust_remote_code=True,
            )
            self.processor = Processor(
                model_name_or_path,
                max_n_frames=32,
            )
        
        self.tokenizer = self.processor.tokenizer

        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            config=model_config,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map=device_map,
            # trust_remote_code=True
        )
        
        self.model.eval()

    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.language_model
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)


encoder_registry = {}
class EncodeMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            encoder_registry[cls.ARCHITECTURE] = cls

    @abstractmethod
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes vision data (images or videos) into a tensor representation.

        Args:
            pixel_values (torch.Tensor | List[torch.Tensor]): The input pixel values. 
                - If a tensor, it should be of shape (B, C, H, W) for images or (B, T, C, H, W) for videos.
                - If a list, it will be stacked into a tensor.

        Returns:
            torch.Tensor: The encoded tensor representation of the input vision data.

        Raises:
            ValueError: If `pixel_values` is not 4D or 5D.

        ## Notes:
            - This function does not accept unbatched inputs.
            - `pixel_values` should be of type uint8.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        """
        Encodes the given text(s) into a tensor representation using the model.

        Args:
            text (str | List[str]): A single string or a list of strings to be encoded.

        Returns:
            torch.Tensor: The tensor representation of the encoded text(s).

        ## Notes:
            - The method uses a prompt to encode the text.
            - If a single string is provided, it is converted into a list containing that string.
            - The method processes the prompts and generates the tensor representation using the model.
            - The output tensor contains the hidden states of the last token for each input text.
        """
        raise NotImplementedError


class TARA(BaseModelForTARA, EncodeMixin):

    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor], edit_text: Optional[str] = None) -> torch.Tensor:

        pixel_values = transform_pixel_values(pixel_values) # [B, T, C, H, W]
        nframes = pixel_values.shape[1]
        
        if edit_text is not None:
            # For composed video retrieval, we need to embed a video with the given text
            prompt = self.video_edit_eol_prompt.replace('<sent>', edit_text)
        else:
            prompt = self.image_eol_prompt if nframes == 1 else self.video_eol_prompt
        
        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)

        generate_kwargs = {
            "max_new_tokens": 1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        vision_embs = []

        for frames in batched_frames:
            input_prompt = prompt.replace("<video>", "<image>"*len(frames))
            input_ids = self.processor.get_text_inputs(input_prompt)
            frames = self.processor.get_pixel_values(frames)
            inputs = {
                "input_ids": input_ids,
                "pixel_values": frames
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            vision_embs.append(outputs.hidden_states[0][-1][:, -1, :])
        
        vision_embs = torch.cat(vision_embs)
        return vision_embs
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        
        prompts = [prompt.replace('<sent>', t) for t in text]

        generate_kwargs = {
            "max_new_tokens": 1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        text_embs = []

        for p in prompts:
            text_inputs = self.processor.get_text_inputs(p)
            inputs = {
                "input_ids": text_inputs,
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            text_embs.append(outputs.hidden_states[0][-1][:, -1, :])
        
        text_embs = torch.cat(text_embs)
        return text_embs

    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:

        pixel_values = transform_pixel_values(pixel_values) # [B, T, C, H, W]
        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)
        descriptions = []
        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": 2048,
            "top_p": 1,
            "temperature": 0,
            "use_cache": True
        }

        for frames in batched_frames:
            text_inputs = f"<video>\n{self.describe_prompt}"
            text_inputs = self.processor.process_prompt(text_inputs, frames)
            text_inputs = self.processor.get_text_inputs(text_inputs)
            frames = self.processor.get_pixel_values(frames)
            inputs = {
                "input_ids": text_inputs,
                "pixel_values": frames
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            output_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
            descriptions.append(output_text)
        return descriptions


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except (ValueError, IndexError):
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(
        video_path, num_frames, sample='middle', fix_start=None, 
        max_num_frames=-1, trimmed30=False, height=-1, width=-1
    ):
    decord.bridge.set_bridge('torch')

    # num_threads = 1 if video_path.endswith('.webm') else 0 # make ssv2 happy
    num_threads = 1
    video_reader = VideoReader(video_path, num_threads=num_threads, height=height, width=width)
    try:
        vlen = len(video_reader)

        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)

        # only use top 30 seconds
        if trimmed30 and duration > 30:
            duration = 30
            vlen = int(30 * float(fps))

        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )

        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames.asnumpy())
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        return frames
    finally:
        # Explicitly release underlying resources to avoid file descriptor leaks
        del video_reader


import PIL.Image
def read_image_decord(image_path):
    image = PIL.Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image


def read_images_decord(image_paths):
    images = []
    for image_path in image_paths:
        image = read_image_decord(image_path)
        images.append(image)
    images = torch.cat(images)
    return images


if __name__ == "__main__":

    # Load model
    model = TARA.from_pretrained(
        "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint",
        device_map='auto',
        dtype=torch.bfloat16,
    )
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Number of parameters: {round(n_params/1e9, 3)}B")
    
    # Let's encode a sample video
    print(colored("Testing video encoding...", 'cyan'))
    video_path = "./assets/folding_paper.mp4"
    video_tensor = read_frames_decord(video_path, num_frames=16)
    video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.to(model.model.device)
    with torch.no_grad():
        video_emb = model.encode_vision(video_tensor).cpu().squeeze(0).float()
    print("Video shape:", video_tensor.shape) # torch.Size([1, 16, 3, 240, 426])
    print("Video embedding shape:", video_emb.shape) # torch.Size([4096])
    
    # Let's encode a sample text
    print(colored("Testing text encoding...", 'cyan'))
    text = ['someone is folding a paper', 'cutting a paper', 'someone is unfolding a paper']
    # NOTE: It can also take a single string
    with torch.no_grad():
        text_emb = model.encode_text(text).cpu().float()
    print("Text:", text)
    print("Text embedding shape:", text_emb.shape) # torch.Size([3, 4096])

