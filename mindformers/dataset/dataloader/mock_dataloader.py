# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Base Mock Dataloader."""

import numpy as np


class BaseMockDataLoader:
    """
    A base class for creating a mock data loader that generates synthetic (dummy) data.
    This is useful for testing or debugging model pipelines without requiring real datasets.

    The loader creates constant tensors (filled with ones) based on specified column names,
    shapes, and data types. All samples returned are identical.
    """

    # Class-level attributes (defaults, to be overridden by instance initialization)
    # List of column names (e.g., ['input_ids', 'labels'])
    mock_columns = None

    # List of shapes for each column (e.g., [[1, 4096], [1, 4096]])
    mock_shapes = None

    # List of data type strings (e.g., ['float32', 'int64'])
    mock_dtypes = None

    # Total number of samples in the dataset (default size = 10240)
    mock_size = 10240

    def __init__(
            self,
            mock_columns: list[str] = None,
            mock_shapes: list[list] = None,
            mock_dtypes: list[str] = None,
            mock_size: int = 10240,
            **kwargs
    ):
        """
        Initializes the mock data loader with given configuration.
        """
        self.mock_columns = mock_columns
        self.mock_shapes = mock_shapes
        self.mock_dtypes = mock_dtypes
        self.mock_size = mock_size

        self.build()

    def build(self):
        """
        Constructs mock tensors (arrays filled with ones) for each column based on the provided
        shapes and data types, and attaches them as instance attributes.
        """
        if not len(self.mock_columns) == len(self.mock_shapes) == len(self.mock_dtypes):
            raise ValueError("mock columns, shapes and dtypes length should be the same.")

        for col, shape, dtypes in zip(self.mock_columns, self.mock_shapes, self.mock_dtypes):
            setattr(self, col, np.ones(tuple(shape), dtype=getattr(np, dtypes)))

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset. Since this is mock data, every sample is identical.

        Args:
            idx (int): Index of the sample (ignored, as all samples are the same).

        Returns:
            list: A list of tensors corresponding to each column, in the order of mock_columns.
        """
        return [getattr(self, col) for col in self.mock_columns]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The size of the dataset (i.e., mock_size).
        """
        return self.mock_size
