import os
import re

# Define placeholders for dataset paths
BMAD_VQA = {
    "annotation_path": "./data/bmad_vqa.json",
    "data_path": "/home/work/data/woohyeon",
}

VQA_RAD = {
    "annotation_path": "./data/vqa_rad_format.json",
    "data_path": "/home/work/data/woohyeon/med_vqa/VQA_RAD/images"
}

PATH_VQA = {
    "annotation_path": "./data/pathvqa_format.json",
    "data_path": "/home/work/data/woohyeon/med_vqa/PathVQA/images"
}

SLAKE_VQA = {
    "annotation_path": "./data/slake_format.json",
    "data_path": "/home/work/data/woohyeon/med_vqa/Slake/images"
}

ANOMALY_SHAPENET_ZERO_SHOT2 = {
    "annotation_path": "./data/anomaly_shapenet_zero_shot2.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

BMAD_ZERO_SHOT = {
    "annotation_path": "./data/bmad_zero_shot.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

MVTEC_ZERO_SHOT = {
    "annotation_path": "./data/mvtec_zero_shot.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

MVTEC3D_ZERO_SHOT2 = {
    "annotation_path": "./data/mvtec3d_zero_shot2.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

REAL3D_ZERO_SHOT2 = {
    "annotation_path": "./data/real3d_zero_shot2.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

WEBAD_PRECESSED = {
    "annotation_path": "./data/webad_processed.json",
    "data_path": "/home/work/data/woohyeon/anomaly_dataset"
}

MIMIC_DIFF_VQA = {
    "annotation_path": "./data/mimic_diff_vqa_train.json",
    "data_path": "/home/work/data/woohyeon"
}

MIMIC_CXR_VQA = {
    "annotation_path": "./data/mimic_cxr_vqa_train.json",
    "data_path": "/home/work/data/woohyeon"
}

MED_ANOMALY_SEG = {
    "annotation_path": "./data/med_anomaly_seg.json",
    "data_path": "/home/work/data/woohyeon"
}

MED_ANOMALY_ONLY_SEG = {
    "annotation_path": "./data/med_anomaly_only_seg.json",
    "data_path": "/home/work/data/woohyeon"
}

CHESTXDET = {
    "annotation_path": "./data/chestxdet.json",
    "data_path": "/home/work/data/woohyeon"
}

# Portable, environment-configured dataset used only for the Stage 1
# functional Training Smoke Test. Keeping the paths outside the repository
# prevents checkpoints and medical images from being committed accidentally.
MEDIC_AD_STAGE1_SMOKE = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_SMOKE_ANNOTATION",
        "/home/data/medic-ad/training-smoke/stage1_smoke.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_SMOKE_DATA_ROOT",
        "/home/data/medic-ad/training-smoke/images",
    ),
}

MEDIC_AD_TINY_VQARAD_TRAIN = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_TINY_VQARAD_TRAIN_ANNOTATION",
        "/home/data/medic-ad/training-tiny-vqarad/train.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_TINY_VQARAD_IMAGE_ROOT",
        "/home/data/medic-ad/training-tiny-vqarad/images",
    ),
}

MEDIC_AD_TINY_VQARAD_VALIDATION = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_TINY_VQARAD_VALIDATION_ANNOTATION",
        "/home/data/medic-ad/training-tiny-vqarad/validation.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_TINY_VQARAD_IMAGE_ROOT",
        "/home/data/medic-ad/training-tiny-vqarad/images",
    ),
}

MEDIC_AD_BASELINE_VQARAD_TRAIN = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_BASELINE_VQARAD_TRAIN_ANNOTATION",
        "/home/data/medic-ad/baseline-vqarad-v1/train.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_BASELINE_VQARAD_IMAGE_ROOT",
        "/home/data/medic-ad/baseline-vqarad-v1/images",
    ),
}

MEDIC_AD_BASELINE_VQARAD_VALIDATION = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_BASELINE_VQARAD_VALIDATION_ANNOTATION",
        "/home/data/medic-ad/baseline-vqarad-v1/validation.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_BASELINE_VQARAD_IMAGE_ROOT",
        "/home/data/medic-ad/baseline-vqarad-v1/images",
    ),
}

MEDIC_AD_STAGE2H_TRAIN = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_STAGE2H_TRAIN_ANNOTATION",
        "/home/data/medic-ad/stage2h-lad-mil-v2/train.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_STAGE2H_IMAGE_ROOT",
        "/home/data/medic-ad/official/med_anomaly",
    ),
}

MEDIC_AD_STAGE2H_CALIBRATION = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION",
        "/home/data/medic-ad/stage2h-lad-mil-v2/calibration.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_STAGE2H_IMAGE_ROOT",
        "/home/data/medic-ad/official/med_anomaly",
    ),
}

MEDIC_AD_STAGE2H_INTERNAL_TEST = {
    "annotation_path": os.environ.get(
        "MEDIC_AD_STAGE2H_INTERNAL_TEST_ANNOTATION",
        "/home/data/medic-ad/stage2h-lad-mil-v2/internal-test.json",
    ),
    "data_path": os.environ.get(
        "MEDIC_AD_STAGE2H_IMAGE_ROOT",
        "/home/data/medic-ad/official/med_anomaly",
    ),
}

data_dict = {
    "bmad_vqa": BMAD_VQA,
    "vqa_rad": VQA_RAD,
    "path_vqa": PATH_VQA,
    "slake_vqa": SLAKE_VQA,
    "anomaly_shapenet_zero_shot2": ANOMALY_SHAPENET_ZERO_SHOT2,
    "bmad_zero_shot": BMAD_ZERO_SHOT,
    "mvtec_zero_shot": MVTEC_ZERO_SHOT,
    "mvtec3d_zero_shot2": MVTEC3D_ZERO_SHOT2,
    "real3d_zero_shot2": REAL3D_ZERO_SHOT2,
    "webad_processed": WEBAD_PRECESSED,
    "mimic_diff_vqa": MIMIC_DIFF_VQA,
    "mimic_cxr_vqa": MIMIC_CXR_VQA,
    "med_anomaly_seg" : MED_ANOMALY_SEG,
    "med_anomaly_only_seg" : MED_ANOMALY_ONLY_SEG,
    "chestxdet" : CHESTXDET,
    "medic_ad_stage1_smoke": MEDIC_AD_STAGE1_SMOKE,
    "medic_ad_tiny_vqarad_train": MEDIC_AD_TINY_VQARAD_TRAIN,
    "medic_ad_tiny_vqarad_validation": MEDIC_AD_TINY_VQARAD_VALIDATION,
    "medic_ad_baseline_vqarad_train": MEDIC_AD_BASELINE_VQARAD_TRAIN,
    "medic_ad_baseline_vqarad_validation": MEDIC_AD_BASELINE_VQARAD_VALIDATION,
    "medic_ad_stage2h_train": MEDIC_AD_STAGE2H_TRAIN,
    "medic_ad_stage2h_calibration": MEDIC_AD_STAGE2H_CALIBRATION,
    "medic_ad_stage2h_internal_test": MEDIC_AD_STAGE2H_INTERNAL_TEST,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
