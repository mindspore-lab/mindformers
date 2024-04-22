from mindspore import Tensor


def get_batch_seq_parallel(batch: dict):
    """
    Args:
        batch (dict): Dictionary containing batch data.

    Returns:
        dict: Transformed batch data to support sequence parallelism.
    """

    # sp_size = get_args().sequence_parallel_size
    sp_size = 0
    if sp_size > 1:
        # sp_rank = get_sequence_parallel_rank()
        sp_rank = 0
        for key, val in batch.items():
            seq_dim = 2 if key == "attention_mask" else 1
            val = val.view(
                *val.shape[0:seq_dim],
                2 * sp_size,
                val.shape[seq_dim] // (2 * sp_size),
                *val.shape[(seq_dim + 1):],
            )
            index = Tensor([sp_rank, (2 * sp_size - sp_rank - 1)])
            val = val.index_select(seq_dim, index)
            val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            batch[key] = val

    return batch
