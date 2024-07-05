"""
AttentionMetadata for pefill mix decode.
"""
class AttentionMetadata:
    """
    AttentionMetadata for pefill mix decode.
    """

    def __init__(self, num_prefill_queries: int, num_decode_queries: int):
        self.num_prefill_querys = num_prefill_queries
        self.num_decode_querys = num_decode_queries
        