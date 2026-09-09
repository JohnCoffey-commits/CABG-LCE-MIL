import copy
import json
import os
import random
import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

try:
    from decord import VideoReader
except Exception:  # pragma: no cover - optional dependency
    VideoReader = None

try:
    from torchcodec.decoders import VideoDecoder
except Exception:  # pragma: no cover - optional dependency
    VideoDecoder = None
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
Image.MAX_IMAGE_PIXELS = None
# Allow unusually large PNG text chunks seen in some medical datasets.
PngImagePlugin.MAX_TEXT_CHUNK = 16 * 4096 * 4096  # 16MB per decompressed chunk
PngImagePlugin.MAX_TEXT_MEMORY = 64 * PngImagePlugin.MAX_TEXT_CHUNK

local_rank = None


def is_main_process() -> bool:
    return local_rank in (None, -1, 0)


def rank0_print(*args):
    if is_main_process():
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image=None,
    grid_thw_video=None,
    use_anomaly_token=False,
    num_pooling_size=2,
    use_diff_token=False,
    diff_only_mode=False,
) -> Dict:
    grid_thw_image = grid_thw_image or []
    grid_thw_video = grid_thw_video or []
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for source in sources:
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except Exception:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if use_anomaly_token and not use_diff_token and not diff_only_mode:
                            replacement = (
                                "<|vision_start|>"
                                + f"<|image_pad|>"
                                * (grid_thw_image[visual_replicate_index_image])
                                + "<|vision_end|>"
                                + "(Anomaly tokens:"
                                + f"<|image_pad|>"
                                * (int(num_pooling_size**2))
                                + ")"
                            )
                        elif use_anomaly_token and use_diff_token and not diff_only_mode:
                            replacement = (
                                "<|vision_start|>"
                                + f"<|image_pad|>"
                                * (grid_thw_image[visual_replicate_index_image])
                                + "<|vision_end|>"
                                + "(Anomaly tokens:"
                                + f"<|image_pad|>"
                                * (int(num_pooling_size**2))
                                + ")"
                            )
                        else:
                            replacement = (
                                "<|vision_start|>"
                                + f"<|image_pad|>"
                                * grid_thw_image[visual_replicate_index_image]
                                + "<|vision_end|>"
                            )

                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    if use_diff_token:
                        diff_prompt = (
                            "(Diff tokens:"
                            + f"<|image_pad|>"
                            * (int(num_pooling_size**2))
                            + ")"
                        )
                        new_parts.append(diff_prompt)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        use_anomaly_token=False,
        num_pooling_size=2,
        dataset_role="training",
        shuffle=True,
    ):
        super(LazySupervisedDataset, self).__init__()

        self.use_anomaly_token = use_anomaly_token
        self.num_pooling_size = num_pooling_size
        self.load_masks = getattr(data_args, "load_masks", True)
        self.require_anomaly_labels = bool(getattr(data_args, "require_anomaly_labels", False))
        self.strict_sample_loading = bool(getattr(data_args, "strict_sample_loading", False))
        self.use_diff_token = getattr(data_args, "use_diff_token", False)
        self.diff_only_mode = getattr(data_args, "diff_only_mode", False)
        if self.diff_only_mode:
            self.use_diff_token = True
        if self.use_diff_token and not self.use_anomaly_token and not self.diff_only_mode:
            raise ValueError("Diff token mode requires anomaly tokens to be enabled.")

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                with open(data["annotation_path"], "r") as f:
                    annotations = json.load(f)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"Sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"Dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total {dataset_role} samples: {len(list_data_dict)}")

        if shuffle:
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            rank0_print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        if VideoReader is not None:
            while decord_attempts < max_decord_attempts:
                try:
                    decord_video = self.video_decord(video_file)
                    return decord_video
                except Exception as e:
                    print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                    decord_attempts += 1
        else:
            rank0_print("Decord is not available; skipping Decord decoding.")

        if VideoDecoder is not None:
            try:
                torchcodec_video = self.video_torchcodec(video_file)
                return torchcodec_video
            except Exception as e:
                print(f"torchcodec attempt failed: {e}")
        else:
            rank0_print("TorchCodec is not available; skipping TorchCodec decoding.")

    def video_decord(self, video_file):
        if VideoReader is None:
            raise ImportError("Decord support is not available in this environment.")
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        if VideoDecoder is None:
            raise ImportError("TorchCodec support is not available in this environment.")
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.strict_sample_loading:
            return self._get_item(i)
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Unexpected nested sample structure."

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None
        mask_tensor = None
        image_tensors: List[torch.Tensor] = []
        grid_thw_list: List[torch.Tensor] = []

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image_outputs, grid_thw = zip(*results)
                    image_tensors = [img for img in image_outputs]
                    grid_thw_list = [thw for thw in grid_thw]
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image_tensor, grid = self.process_image_unified(image_file)
                    image_tensors = [image_tensor]
                    grid_thw_list = [grid]
            else:
                image_file = os.path.join(image_folder, image_file)
                image_tensor, grid = self.process_image_unified(image_file)
                image_tensors = [image_tensor]
                grid_thw_list = [grid]
            grid_thw_merged = copy.deepcopy(grid_thw_list)
            if not isinstance(grid_thw_list, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw_list = [grid_thw_list]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            grid_thw = grid_thw_list
        def _images_in_tensor(tensor: torch.Tensor) -> int:
            # For flattened patch tensors (<=3 dims) treat the whole tensor as one image.
            return 1 if tensor.dim() <= 3 else tensor.shape[0]

        num_images = sum(_images_in_tensor(img_tensor) for img_tensor in image_tensors)
        anomaly_label = None
        if self.require_anomaly_labels:
            if num_images != 1:
                raise RuntimeError(
                    f"Stage 2H requires exactly one image per sample, got {num_images}."
                )
            anomaly_label = sources[0].get("anomaly_label")
            if isinstance(anomaly_label, bool) or not isinstance(anomaly_label, int):
                raise RuntimeError("Stage 2H anomaly_label must be an explicit integer 0 or 1.")
            if anomaly_label not in (0, 1):
                raise RuntimeError("Stage 2H anomaly_label must be 0 or 1.")
        if self.use_diff_token:
            if num_images != 2:
                raise RuntimeError("Diff mode requires exactly two images per sample.")

        # Load segmentation masks when requested.
        mask_tensor = None
        if self.load_masks:
            if "mask" in sources[0]:
                mask_folder = self.list_data_dict[i]["data_path"]
                mask_file = self.list_data_dict[i]["mask"]

                full_mask_path = os.path.join(mask_folder, mask_file)
                mask = Image.open(full_mask_path).convert("L")

                target_size = (512, 512)  # (width, height)
                mask_resized = mask.resize(target_size, Image.Resampling.NEAREST)

                mask_np = np.array(mask_resized)
                mask_np = (mask_np != 0).astype(np.uint8)

                mask_tensor = torch.from_numpy(mask_np).to(torch.uint8).unsqueeze(0)
            else:
                mask_tensor = torch.zeros((512, 512), dtype=torch.uint8).unsqueeze(0)

            if mask_tensor is not None:
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                if num_images > 0:
                    if mask_tensor.shape[0] == 1 and num_images > 1:
                        mask_tensor = mask_tensor.repeat(num_images, 1, 1)
                    elif mask_tensor.shape[0] != num_images:
                        raise RuntimeError(
                            f"Mask tensor count ({mask_tensor.shape[0]}) does not match number of images ({num_images})."
                        )
                elif num_images == 0:
                    mask_tensor = None

        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, list):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
            use_anomaly_token=self.use_anomaly_token,
            num_pooling_size=self.num_pooling_size,
            use_diff_token=self.use_diff_token,
            diff_only_mode=self.diff_only_mode,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw_image=grid_thw_merged,
                use_anomaly_token=self.use_anomaly_token,
                num_pooling_size=self.num_pooling_size,
                use_diff_token=self.use_diff_token,
                diff_only_mode=self.diff_only_mode,
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        if image_tensors:
            data_dict["pixel_values"] = torch.cat(image_tensors, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw_list], dim=0
            )
        if self.load_masks and mask_tensor is not None:
            data_dict["masks"] = mask_tensor
        else:
            data_dict["masks"] = None
        if self.require_anomaly_labels:
            data_dict["anomaly_labels"] = torch.tensor([anomaly_label], dtype=torch.long)
        # video exist in the data
        if "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    load_masks: bool = True
    require_anomaly_labels: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        anomaly_label_chunks = []
        if self.require_anomaly_labels:
            for index, instance in enumerate(instances):
                image_grid = instance.get("image_grid_thw")
                image_tensor = instance.get("pixel_values")
                labels_for_image = instance.get("anomaly_labels")
                if image_grid is None or image_tensor is None:
                    raise RuntimeError(f"Stage 2H sample {index} is missing its image tensors.")
                image_count = int(image_grid.shape[0])
                if image_count != 1:
                    raise RuntimeError(
                        f"Stage 2H sample {index} must contain exactly one image, got {image_count}."
                    )
                if labels_for_image is None:
                    raise RuntimeError(f"Stage 2H sample {index} is missing anomaly_labels.")
                labels_for_image = labels_for_image.reshape(-1)
                if labels_for_image.numel() != image_count:
                    raise RuntimeError(
                        f"Stage 2H sample {index} label/image mismatch: "
                        f"{labels_for_image.numel()} labels vs {image_count} images."
                    )
                if labels_for_image.dtype == torch.bool or labels_for_image.is_floating_point():
                    raise RuntimeError("Stage 2H anomaly_labels must be integer tensors.")
                if not torch.all((labels_for_image == 0) | (labels_for_image == 1)):
                    raise RuntimeError("Stage 2H anomaly_labels contain a value outside {0,1}.")
                anomaly_label_chunks.append(labels_for_image.to(dtype=torch.long))
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        concat_masks = None
        image_counts: List[int] = []
        if self.load_masks:
            mask_chunks: List[torch.Tensor] = []
            for instance in instances:
                mask = instance.get("masks", None)
                image_tensor = instance.get("pixel_values", None)
                image_grid = instance.get("image_grid_thw", None)
                if image_grid is not None:
                    image_count = int(image_grid.shape[0])
                elif image_tensor is not None:
                    image_count = 1 if image_tensor.dim() <= 3 else int(image_tensor.shape[0])
                else:
                    image_count = 0
                image_counts.append(image_count)
                if image_count == 0 and mask is None:
                    continue
                if mask is None:
                    if image_tensor is None:
                        continue
                    if image_tensor.dim() >= 3:
                        height, width = image_tensor.shape[-2:]
                    else:
                        height = width = 512
                    mask = torch.zeros(
                        (image_count, height, width),
                        dtype=torch.uint8,
                    )
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                if image_count > 0 and mask.shape[0] == 1 and image_count > 1:
                    mask = mask.repeat(image_count, 1, 1)
                mask_chunks.append(mask)
            if len(mask_chunks) != 0:
                concat_masks = torch.cat(mask_chunks, dim=0)

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        if self.require_anomaly_labels:
            batch["anomaly_labels"] = torch.cat(anomaly_label_chunks, dim=0)
            if grid_thw is None or batch["anomaly_labels"].numel() != int(grid_thw.shape[0]):
                raise RuntimeError("Stage 2H collated anomaly-label/image alignment failed.")
        if self.load_masks:
            if concat_masks is None and concat_images is not None:
                total_images = sum(image_counts)
                if total_images > 0:
                    concat_masks = torch.zeros(
                        (total_images, 512, 512), dtype=torch.uint8
                    )
            batch["masks"] = concat_masks
        else:
            batch["masks"] = None

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer
    load_masks: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = torch.ones_like(input_ids[0], dtype=torch.bool)

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=cumsum_seq_lens,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        batch["masks"] = None  # Masks are not collated in flattened mode.

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, use_anomaly_token=False, num_pooling_size=2
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        use_anomaly_token=use_anomaly_token,
        num_pooling_size=num_pooling_size,
        dataset_role="training",
        shuffle=True,
    )
    eval_dataset = None
    eval_dataset_use = str(getattr(data_args, "eval_dataset_use", "") or "").strip()
    if eval_dataset_use:
        eval_data_args = copy.copy(data_args)
        eval_data_args.dataset_use = eval_dataset_use
        eval_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_args=eval_data_args,
            use_anomaly_token=use_anomaly_token,
            num_pooling_size=num_pooling_size,
            dataset_role="evaluation",
            shuffle=False,
        )
    require_anomaly_labels = bool(getattr(data_args, "require_anomaly_labels", False))
    if data_args.data_flatten:
        if require_anomaly_labels:
            raise RuntimeError("Stage 2H does not permit flattened/packed collation.")
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer, load_masks=getattr(data_args, "load_masks", True))
        return dict(
            train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        load_masks=getattr(data_args, "load_masks", True),
        require_anomaly_labels=require_anomaly_labels,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )
