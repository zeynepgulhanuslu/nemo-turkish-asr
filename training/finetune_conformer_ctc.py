import argparse
import copy
import logging
import os
from collections import defaultdict
import pytorch_lightning as ptl
from nemo.collections.asr.models import EncDecCTCModel,EncDecCTCModelBPE
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
    trainer = pl.Trainer(**cfg.trainer)

    asr_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    asr_model.change_vocabulary(new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type=cfg.model.tokenizer.type)

    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)
    asr_model.setup_optimization(optim_config=cfg.model.optim)


    # Initialize the weights of the model from another model, if provided via config
    #asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)
    #asr_model.save_to('conformer_smaii_128_hi.nemo')

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
