# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test whisper english_normalizer"""

from unittest.mock import patch
import pytest
from mindformers.models.whisper.english_normalizer import (
    remove_symbols_and_diacritics,
    remove_symbols,
    BasicTextNormalizer,
    EnglishNumberNormalizer,
    EnglishTextNormalizer,
    EnglishSpellingNormalizer
)


# pylint: disable=W0613
def mock_normalize(a, b):
    return b


def mock_category(string):
    return string


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@patch("unicodedata.category", mock_category)
@patch("unicodedata.normalize", mock_normalize)
def test_remove_symbols_and_diacritics():
    """
    Feature: whisper.english_normalizer.remove_symbols_and_diacritics
    Description: test remove_symbols_and_diacritics logic
    Expectation: Success
    """
    assert remove_symbols_and_diacritics("mock", keep="mock") == "mock"
    assert remove_symbols_and_diacritics("Ł") == "L"
    assert not remove_symbols_and_diacritics(["Mn"])
    assert remove_symbols_and_diacritics(["MSP"]) == " "


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_remove_symbols():
    """
    Feature: whisper.english_normalizer.remove_symbols
    Description: test remove_symbols logic
    Expectation: Success
    """
    assert remove_symbols("mock") == "mock"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_basic_text_normalizer():
    """
    Feature: whisper.english_normalizer.BasicTextNormalizer
    Description: test BasicTextNormalizer logic
    Expectation: Success
    """
    basic_text_normalizer = BasicTextNormalizer()
    assert basic_text_normalizer("mock") == "mock"
    basic_text_normalizer = BasicTextNormalizer(split_letters=True)
    assert basic_text_normalizer("mock") == "m o c k"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_english_number_normalizer():
    """
    Feature: whisper.english_normalizer.EnglishNumberNormalizer
    Description: test EnglishNumberNormalizer logic
    Expectation: Success
    """
    english_number_normalizer = EnglishNumberNormalizer()
    assert english_number_normalizer("one and a half two") == 'one.52'
    assert english_number_normalizer("mock") == "mock"
    assert english_number_normalizer("An increasing sequence: one, two, three, five, six, seven, nine.") == \
           'An increasing sequence: one, two, three, five, six, seven, nine.'
    assert english_number_normalizer("5 and a half 2") == '5 and a half 2'
    assert english_number_normalizer("sixes and a half 2") == '6s and a half 2'
    assert english_number_normalizer("fifteen and a half fifteen") == '15.515'
    assert english_number_normalizer("15") == english_number_normalizer("fifteen") == "15"
    assert english_number_normalizer("hundred and a half fifteen") == '100.515'
    assert english_number_normalizer("hundred and a half hundred") == '10050'
    assert english_number_normalizer("twenty and a half twenty") == '20.520'
    assert english_number_normalizer("twenties and a half twenties") == '20s and a half 20s'
    assert english_number_normalizer("twentieth and a half twentieth") == '20th and a half 20th'
    assert english_number_normalizer("five hundred") == '500'
    assert english_number_normalizer("one hundred and fifty") == '150'
    assert english_number_normalizer("10 double two") == '1022'
    assert english_number_normalizer("10 double fifty") == '10 double 50'
    assert english_number_normalizer("hundreds and fifty") == '100s and 50'


def test_english_spelling_normalizer():
    """
    Feature: whisper.english_normalizer.EnglishSpellingNormalizer
    Description: test EnglishSpellingNormalizer logic
    Expectation: Success
    """
    english_spelling_normalizer = EnglishSpellingNormalizer({"mock": "mock_re"})
    assert english_spelling_normalizer("mock") == "mock_re"


def test_english_text_normalizer():
    """
    Feature: whisper.english_normalizer.EnglishTextNormalizer
    Description: test EnglishTextNormalizer logic
    Expectation: Success
    """
    english_text_normalizer = EnglishTextNormalizer({"mock": "mock_re"})
    assert english_text_normalizer("mock") == "mock_re"
