# train model from scratch
import argparse

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf


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

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    print('training starts here')
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':

    main()
