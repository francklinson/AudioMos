import os
from pathlib import Path

# 优先使用项目目录下的模型
# 路径: utmos/utmosv2/utils/_constants.py -> 项目根目录
# 计算项目根目录: app/algorithms/utmos/utmosv2/utils/ -> 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.parent  # AudioMos目录
PROJECT_MODEL_PATH = PROJECT_ROOT / "models" / "utmos"

# UTMOS期望的路径结构是 utmosv2/models/fusion_stage3/
# 所以我们需要在models/utmos下创建models子目录的链接或复制文件
PROJECT_MODEL_PATH_WITH_SUBDIR = PROJECT_MODEL_PATH / "models"

if PROJECT_MODEL_PATH_WITH_SUBDIR.exists():
    _UTMOSV2_CHACHE = PROJECT_MODEL_PATH
    print(f"[UTMOS] 使用项目本地模型路径: {_UTMOSV2_CHACHE}")
elif PROJECT_MODEL_PATH.exists():
    # 如果模型直接放在models/utmos/下，使用这个路径
    _UTMOSV2_CHACHE = PROJECT_MODEL_PATH
    print(f"[UTMOS] 使用项目本地模型路径: {_UTMOSV2_CHACHE}")
else:
    _UTMOSV2_CHACHE = Path(os.getenv("UTMOSV2_CHACHE", "~/.cache/utmosv2")).expanduser()
    print(f"[UTMOS] 使用默认缓存路径: {_UTMOSV2_CHACHE}")
