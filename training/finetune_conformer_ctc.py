import logging

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from dataloader.manifest_util import read_manifest, get_charset


@hydra_runner(config_path="../conf", config_name="conformer_ctc_char")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_medium")

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    asr_model.set_trainer(trainer)
    print('training starts here')
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
