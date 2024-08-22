"""MindFormers DataHandler."""
from mindformers.dataset.handler.build_data_handler import build_data_handler
from mindformers.dataset.handler.alpaca_handler import AlpacaInstructDataHandler
from mindformers.dataset.handler.deepseek_handler import DeepSeekInstructDataHandler

__all__ = ["build_data_handler", "AlpacaInstructDataHandler", "DeepSeekInstructDataHandler"]
