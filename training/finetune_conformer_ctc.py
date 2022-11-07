import argparse
import copy
import logging
import os
from collections import defaultdict
import pytorch_lightning as ptl
import nemo.collections.asr as nemo_asr
import torch
from nemo.utils import exp_manager
from omegaconf import open_dict, OmegaConf
from torch import nn

from dataloader.manifest_util import read_manifest, get_charset, write_processed_manifest
from training.finetune_default_quartznet import remove_special_characters, apply_preprocessors

@hydra_runner(config_path="../conf", config_name="conformer_ctc_bpe_finetune")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    train_manifest_data = read_manifest(str(cfg.model.train_ds.manifest_filepath))
    train_charset = get_charset(train_manifest_data)

    dev_manifest_data = read_manifest(str(cfg.model.validation_ds.manifest_filepath))
    dev_charset = get_charset(dev_manifest_data)
    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))

    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    print('training starts here')
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()



if __name__ == '__main__':
    main()
