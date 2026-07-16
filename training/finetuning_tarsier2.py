import os
import sys
import warnings

# Make the TARA repo root importable (modeling_tara, tarsier2, shared, ...)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Make sibling training modules importable (finetuning.py) regardless of CWD.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
import transformers
from transformers import set_seed
from modeling_tara import BaseModelForTARA
from tarsier2.dataset.utils import format_one_sample
from finetuning import DataCollatorForSeq2SeqForNeg, Similarity, generate_sentemb_prompt, SentembTrainer

warnings.simplefilter(action='ignore', category=FutureWarning)


class SentembTrainerForTarsier2(SentembTrainer):
    @staticmethod
    def _build_text_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids.unsqueeze(-1).expand(-1, -1, 3)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.is_nli and self.use_neg_sentence:
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs["attention_mask"]
            pad_token_id = self.tokenizer.pad_token_id
            if self.fix_attention_mask:
                labels[labels < 0] = pad_token_id
                neg[neg < 0] = pad_token_id
            else:
                labels[labels < 0] = 0
                neg[neg < 0] = 0
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))

            pad_size = mw - labels.size(1)
            if pad_size > 0:
                label_pads = torch.zeros(labels.size(0), pad_size, device=input_ids.device).long()
                label_pads.fill_(pad_token_id)
                labels = torch.cat([label_pads, labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_pads = torch.zeros(input_ids.size(0), pad_size, device=input_ids.device).long()
                input_pads.fill_(pad_token_id)
                input_ids = torch.cat([input_pads, input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg_pads = torch.zeros(neg.size(0), pad_size, device=input_ids.device).long()
                neg_pads.fill_(pad_token_id)
                neg = torch.cat([neg_pads, neg], dim=1)

            inputs["input_ids"] = torch.cat([input_ids, labels, neg], dim=0)
            if self.fix_attention_mask:
                inputs["attention_mask"] = (inputs["input_ids"] != pad_token_id).long()
            else:
                inputs["attention_mask"] = (inputs["input_ids"] > 0).long()
            del inputs["labels"]
        elif self.is_nli:
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0] = 0
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size, device=input_ids.device).long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size, device=input_ids.device).long(), input_ids], dim=1)
            inputs["input_ids"] = torch.cat([input_ids, labels], dim=0)
            inputs["attention_mask"] = (inputs["input_ids"] > 0).long()
            del inputs["labels"]
        else:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], inputs["input_ids"]], dim=0)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], inputs["attention_mask"]], dim=0)
            del inputs["labels"]

        inputs["position_ids"] = self._build_text_position_ids(inputs["attention_mask"])
        input_ids_for_embeds = inputs["input_ids"]
        model_inputs = {
            "inputs_embeds": model.get_input_embeddings()(input_ids_for_embeds),
            "attention_mask": inputs["attention_mask"],
            "position_ids": inputs["position_ids"],
            "output_hidden_states": True,
            "return_dict": True,
        }
        hidden_states = model(**model_inputs).hidden_states[-1]
        pooling_strategy = getattr(self, "pooling_strategy", "last_token")
        if pooling_strategy == "last_token":
            pooler_output = hidden_states[:, -1, :]
        elif pooling_strategy == "avg_pool":
            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
            token_counts = attention_mask.sum(dim=1).clamp(min=1.0)
            pooler_output = (hidden_states * attention_mask).sum(dim=1) / token_counts
        else:
            raise ValueError(
                f"Unsupported pooling_strategy: {pooling_strategy}. "
                "Expected one of: ['last_token', 'avg_pool']."
            )

        if self.use_neg_sentence:
            batch_size = pooler_output.size(0) // 3
            pooler_output = torch.stack(
                [pooler_output[:batch_size], pooler_output[batch_size:2 * batch_size], pooler_output[2 * batch_size:]], dim=1
            )
            z1, z2, z3 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]
        else:
            batch_size = pooler_output.size(0) // 2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if self.use_neg_sentence:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(inputs["input_ids"].device)

        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(input_ids.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss


def train(
    model_name_or_path: str = "",
    data_path: str = "data/nli_for_simcse.csv",
    output_dir: str = "./lora-alpaca",
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    warmup_ratio: float = 0.2,
    cutoff_len: int = 32,
    group_by_length: bool = False,
    run_name: str = None,
    use_neg_sentence: bool = False,
    save_steps: int = 10000,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = False,
    fix_attention_mask: bool = False,
    set_pad_to_unk: bool = False,
    bf16: bool = False,
    pooling_strategy: str = "last_token",
    architecture: str = None,
    local_rank: int = 0,
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    base_model = BaseModelForTARA.from_pretrained(model_name_or_path, load_llm=True, device_map="cuda")
    model = base_model.model
    tokenizer = base_model.tokenizer
    if not hasattr(base_model, "super_processor"):
        raise ValueError("This trainer requires a Tarsier2 base model with `super_processor`.")
    super_processor = base_model.super_processor

    if set_pad_to_unk:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    mask_embedding_sentence_template = base_model.text_eol_prompt
    print(mask_embedding_sentence_template)

    def process_prompt(prompt: str):
        sample = format_one_sample(media_file=None, prompt=prompt)
        result = super_processor.processor(
            sample["messages"],
            image_processing_config={},
            is_training=True,
        )
        input_ids = result["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids[0].tolist()
        return input_ids

    def tokenize(prompt, label_prompt=None, neg_prompt=None):
        input_ids = process_prompt(prompt)
        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if label_prompt:
            result["labels"] = process_prompt(label_prompt)
            if neg_prompt:
                # Reuse attention_mask field as negative sequence to preserve existing collator/trainer behavior.
                result["attention_mask"] = process_prompt(neg_prompt)
        else:
            result["labels"] = result["input_ids"].copy()
            result["attention_mask"] = [1] * len(result["input_ids"])
        return result

    def generate_and_tokenize_prompt(data_point):
        data_point["input"] = data_point["sent0"]
        data_point["output"] = data_point["sent1"]
        if use_neg_sentence:
            data_point["neg"] = data_point["hard_neg"]

        full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="input")
        pos_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="output")
        if use_neg_sentence:
            neg_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="neg")
        else:
            neg_full_prompt = None
        return tokenize(full_prompt, label_prompt=pos_full_prompt, neg_prompt=neg_full_prompt)

    if grad_checkpoint:
        model.enable_input_require_grads()

    if "csv" in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=8)
    dc_fun = DataCollatorForSeq2SeqForNeg if use_neg_sentence else transformers.DataCollatorForSeq2Seq
    data_collator = dc_fun(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    trainer = SentembTrainerForTarsier2(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=logging_steps,
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=0,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
            remove_unused_columns=False,
        ),
        data_collator=data_collator,
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = True
    trainer.use_neg_sentence = use_neg_sentence
    trainer.fix_attention_mask = fix_attention_mask
    trainer.pooling_strategy = pooling_strategy
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("Starting training")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
