from mindspore import nn


class BaseModel(nn.Cell):
    def __init__(self, load_checkpoint: bool = False, checkpoint_path: str = None):
        super(BaseModel, self).__init__()
        self.backbone = None
        self.head = None
        self.load_checkpoink = load_checkpoint
        self.checkpoint_path = checkpoint_path

    def load_checkpoint(self):
        if self.load_checkpoink:
            if self.checkpoint_path is not None:
                pass
            else:
                self.pull_checkpoint()

    def pull_checkpoint(self):
        pass
