"""Project-wide constants."""

RELATION_LABELS = ["extends", "reframes", "replaces", "strengthens", "none"]
RELATION2ID = {r: i for i, r in enumerate(RELATION_LABELS)}
ID2RELATION = {i: r for i, r in enumerate(RELATION_LABELS)}

DEFAULT_BACKBONE = "allenai/specter2_base"
DEFAULT_PROJ_DIM = 256
DEFAULT_HIDDEN_DIM = 768
DEFAULT_NUM_BUCKETS = 300_000
DEFAULT_MEMORY_DIM = 512
