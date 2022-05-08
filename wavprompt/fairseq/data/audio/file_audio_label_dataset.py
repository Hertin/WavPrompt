import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset, BaseWrapperDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from .raw_audio_dataset import RawAudioDataset
logger = logging.getLogger(__name__)

class FileAudioLabelDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        n_shot=0,
        scenarios=None,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        skipped = 0
        self.fnames = []
        sizes = []
        self.n_shot = n_shot
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                if scenarios is not None and items[2] not in scenarios:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        import soundfile as sf
        all_feats = []
        for fbname in str(self.fnames[index]).split(';'):
            fname = os.path.join(self.root_dir, fbname)
            wav, curr_sample_rate = sf.read(fname)
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, dim=0)
        return {"id": index, "source": all_feats}

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes), self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes, self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )