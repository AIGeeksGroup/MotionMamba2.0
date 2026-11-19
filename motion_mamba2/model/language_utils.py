# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2024 McGill NLP
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# The above copyright notice and this permission notice shall be included in
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.
'Language utils.'
import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available

def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available('transformers'):
        return False
    return version.parse(importlib.metadata.version('transformers')) >= version.parse('4.43.1')
