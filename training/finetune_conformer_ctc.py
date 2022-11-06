import logging

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from dataloader.manifest_util import read_manifest, get_charset


def main():
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_medium")

    from omegaconf import OmegaConf, open_dict
    params = OmegaConf.load("conformer_ctc_char.yaml")
    print(OmegaConf.to_yaml(params))

    params.model.spec_augment.rect_masks = 0

    import copy
    new_opt = copy.deepcopy(params.model.optim)
    new_opt.lr = 0.001

    print(params.model.optim)

    print(asr_model.decoder.vocabulary)

    asr_model.setup_optimization(optim_config=new_opt)

    asr_model.setup_training_data(train_data_config=params['model']['train_ds'])

    asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

    asr_model.encoder.freeze()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    early_stop_callback = EarlyStopping(
        monitor='val_wer',
        min_delta=0.05,
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=2,
                         accumulate_grad_batches=1,
                         checkpoint_callback=False,
                         logger=False,
                         log_every_n_steps=5,
                         check_val_every_n_epoch=1,
                         callbacks=[early_stop_callback],
                         gpus=0,
                         # accelerator='ddp',
                         # plugins='ddp_sharded'
                         )

    asr_model.set_trainer(trainer)

    trainer.fit(asr_model)


'''
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

'''
if __name__ == '__main__':
    main()
