import math


PROTOCOL_VERSION = "cabg-mil-v1.1"
CHECKPOINT_SCHEMA_VERSION = "cabg-v1.1-state-1"
BLOCK_TRACE_SCHEMA_VERSION = "cabg-v1.1-block-trace-1"
DATASET_SCHEMA_VERSION = "cabg-v1.1-dataset-1"

POSITIONS = 1024
TAU = 1.0 / math.log(POSITIONS)
EMA_BETA = 0.9
RHO = 0.10
RHO_MAX = 0.20
EPSILON = 1e-8
LM_REFERENCE_FLOOR = 1e-12
GLOBAL_CLIP_NORM = 1.0

NORMAL_LABEL = "normal"
ABNORMAL_LABEL = "abnormal"
LABEL_ALIASES = {
    "good": NORMAL_LABEL,
    "normal": NORMAL_LABEL,
    "no": NORMAL_LABEL,
    "ungood": ABNORMAL_LABEL,
    "abnormal": ABNORMAL_LABEL,
    "yes": ABNORMAL_LABEL,
}

SHARED_SUPPORT_NAMES = (
    "model.visual.anomaly_qformer.abnormal_prompt",
    "model.visual.anomaly_qformer.normal_prompt",
    "model.visual.anomaly_qformer.anomaly_attention.query_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.query_proj.bias",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.weight",
    "model.visual.anomaly_qformer.anomaly_attention.key_proj.bias",
)

EXPECTED_SHARED_TENSORS = 6
EXPECTED_SHARED_ELEMENTS = 1_314_304
EXPECTED_TRAINABLE_TENSORS = 21
EXPECTED_TRAINABLE_ELEMENTS = 29_561_345

LOCKED_PROTOCOL_SHA256 = "99e890fa2669c94a953d1f3715d17e96b6ebc272abaf628fda4334a5c755ab0b"
REVIEWED_PROTOCOL_SHA256 = "cef05b0a5bf1a989edc448efb338540484ea2297adbf02b92a60a1218c94650c"
