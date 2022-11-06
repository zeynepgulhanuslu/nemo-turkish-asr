# train model from scratch
import argparse

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from dataloader.manifest_util import read_manifest, get_charset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        help="""Config file
        """,
    )

    return parser.parse_args()


@hydra_runner(config_path="../conf", config_name="conformer_ctc_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    train_manifest_data = read_manifest((cfg.model, 'train_ds'))
    train_charset = get_charset(train_manifest_data)

    dev_manifest_data = read_manifest((cfg.model, 'validation_ds'))
    dev_charset = get_charset(dev_manifest_data)
    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))

    asr_model.cfg.labels = list(train_dev_set)
    asr_model.cfg.validation_ds.labels = list(train_dev_set)
    print('labels:', asr_model.cfg.labels)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    print('training starts here')
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':

    main()
