#!/usr/bin/env bash
# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

set -euo pipefail

echo "Formatting Python files in the repo with isort and black..."
echo

# Format with isort first (import sorting)
isort .

# Then format with black (code style)
black .

echo
echo "✅ Done formatting the repo with isort and black."