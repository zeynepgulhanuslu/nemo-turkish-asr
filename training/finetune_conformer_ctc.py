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
from nemo.core.config import hydra_runner
import pytorch_lightning as pl
from nemo.utils import logging

@hydra_runner(config_path="../conf", config_name="conformer_ctc_bpe_finetune")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)

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
