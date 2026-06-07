"""
ModelScope 兼容性补丁

解决 modelscope==1.37.1 与 datasets>=2.14.0 的兼容性问题：
- datasets.LargeList 在 datasets 2.14.0+ 中被移除
- datasets.features.features._FEATURE_TYPES 在 datasets 2.14.0+ 中被移除

此模块应在任何导入 modelscope 的代码之前导入。
"""

import datasets
import datasets.features.features as _ds_ff


def apply_patches():
    """应用所有兼容性补丁"""

    # 补丁1: LargeList
    if not hasattr(datasets, "LargeList"):
        class _LargeListStub(list):
            """Stub for datasets.LargeList removed in datasets >= 2.14.0."""
            pass

        datasets.LargeList = _LargeListStub

    # 补丁2: _FEATURE_TYPES
    if not hasattr(_ds_ff, "_FEATURE_TYPES"):
        from datasets.features.features import (
            Value,
            ClassLabel,
            Array2D,
            Array3D,
            Array4D,
            Array5D,
        )

        _FEATURE_TYPES = {
            "Value": Value,
            "ClassLabel": ClassLabel,
            "Sequence": datasets.Sequence,  # 使用 datasets.Sequence（非 typing.Sequence）
            "Array2D": Array2D,
            "Array3D": Array3D,
            "Array4D": Array4D,
            "Array5D": Array5D,
        }
        _ds_ff._FEATURE_TYPES = _FEATURE_TYPES


# 自动应用补丁
apply_patches()
