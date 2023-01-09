import logging
import json
import os
from threading import Thread
from PIL import Image
import numpy as np

from mindspore import context, Tensor, Model, nn, load
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision.c_transforms as C

os.environ["GLOG_v"] = "3"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_lens=-1):
    """pad_sequence"""
    lens = [len(x) for x in sequences]
    if max_lens == -1:
        max_lens = max(lens)

    padded_seq = []
    for x in sequences:
        pad_width = [(0, max_lens - len(x))]
        padded_seq.append(np.pad(x, pad_width, constant_values=(padding_value, padding_value)))

    sequences = np.stack(padded_seq, axis=0 if batch_first else 1)
    return sequences


def pad_tensors(tensors, lens=None, pad=0, max_len=-1):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if max_len == -1:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = np.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output


def decode_sequence(ix_to_word, seq, split=' '):
    """
    decode_sequence
    """
    N = seq.shape[0]
    D = seq.shape[1]
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + split
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt.replace(' ##', ''))
    return out


class opt_caption_inference:
    def __init__(self, model_path, model_name, vocab_name):
        self.image_size = 448
        self.patch_size = 32

        resize = self.image_size
        image_size = self.image_size
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        interpolation = "BILINEAR"
        if hasattr(Inter, interpolation):
            interpolation = getattr(Inter, interpolation)
        else:
            interpolation = Inter.BILINEAR
            logger.warning('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))
        self.trans = [
            C.Resize(resize, interpolation=interpolation),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

        self.model_path = model_path
        self.model_name = model_name
        self.vocab_name = vocab_name
        model = os.path.join(self.model_path, self.model_name)
        logger.info(f"load model: {model}")
        self.graph = load(model)
        self.network = nn.GraphCell(self.graph)
        self.model = Model(self.network)
        vocab = os.path.join(self.model_path, self.vocab_name)
        self.vocab = json.load(open(vocab))
        # 模型预热，否则首次推理的时间会很长
        self.load = Thread(target=self._warmup)
        self.load.start()
        logger.info("load network successfully!")

    def _warmup(self):
        from mindspore import float32, int64
        logger.info("warmup network...")
        image = Tensor(np.array(np.random.randn(1, 197, 3072), dtype=np.float32), float32)
        img_pos_feat = Tensor(np.expand_dims(np.arange(0, 197, dtype=np.int64), axis=0), int64)
        attn_masks = Tensor(np.ones((1, 197), dtype=np.int64), int64)
        gather_index = Tensor(np.expand_dims(np.arange(0, 197, dtype=np.int64), axis=0), int64)
        self.model.predict(image, img_pos_feat, attn_masks, gather_index)
        logger.info("warmup network successfully!")

    def preprocess(self, image):
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        for tran in self.trans:
            image = tran(image)

        p = self.patch_size
        channels, h, w = image.shape
        x = np.reshape(image, (channels, h // p, p, w // p, p))
        x = np.transpose(x, (1, 3, 0, 2, 4))
        patches = np.reshape(x, ((h // p) * (w // p), channels * p * p))
        img_pos_feat = np.arange(patches.shape[0] + 1)
        attn_masks = np.ones(img_pos_feat.shape[0], dtype=np.int64)

        img_feat = Tensor(pad_tensors([patches, ], [196], max_len=197))
        img_pos_feat = Tensor(np.stack([img_pos_feat, ], axis=0))
        attn_masks = Tensor(pad_sequence([attn_masks, ], batch_first=True, padding_value=0, max_lens=197))
        out_size = attn_masks.shape[1]
        batch_size = attn_masks.shape[0]
        gather_index = Tensor(np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0))

        return img_feat, img_pos_feat, attn_masks, gather_index

    def postprocess(self, sequence):
        return decode_sequence(self.vocab, sequence[:, 0, 1:].asnumpy(), split='')

    def inference(self, input_data):
        # 阻塞预热
        self.load.join()

        inference_result = {}
        for k, v in input_data.items():
            instance_result = {}
            for file_name, file_content in v.items():
                (image, img_pos_feat, attn_masks, gather_index) = self.preprocess(file_content)
                sequence = self.model.predict(image, img_pos_feat, attn_masks, gather_index)
                instance_result[file_name] = self.postprocess(sequence)
            inference_result[k] = instance_result
        return inference_result
