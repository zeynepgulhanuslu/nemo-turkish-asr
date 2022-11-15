# train model from scratch
import argparse

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel,EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from dataloader.manifest_util import read_manifest, get_charset




@hydra_runner(config_path="../conf", config_name="conformer_ctc_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    train_manifest_data = read_manifest(str(cfg.model.train_ds.manifest_filepath))[0:300000]
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
