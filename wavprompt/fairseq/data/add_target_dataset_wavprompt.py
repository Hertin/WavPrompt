import torch

from . import BaseWrapperDataset, data_utils

class AddTargetDatasetWavPrompt(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        add_to_input=False,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.labels[index])
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [torch.LongTensor(s["label"]) for s in samples if s["id"] in indices]
        attention_mask_raw = [torch.LongTensor([1] * len(t)) for t in target]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.eos, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
            
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["target"] = target

        if self.add_to_input:
            prev_output_tokens = target.clone().long()
            attention_mask = data_utils.collate_tokens(attention_mask_raw, pad_idx=0, left_pad=False)
            assert attention_mask.shape == target.shape, f'attention_mask {attention_mask.shape} != target {target.shape}'
   
            collated["net_input"]["prev_output_tokens"] = prev_output_tokens
            collated["net_input"]["attention_mask"] = attention_mask
            collated["net_input"]["id"] = torch.LongTensor([s["id"] for s in samples if s["id"] in indices])
            collated["ntokens"] += target.size(0)

        return collated
