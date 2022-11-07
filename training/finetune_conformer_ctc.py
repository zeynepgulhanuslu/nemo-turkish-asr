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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manifest', type=str, required=True, help='Manifest directory')
    parser.add_argument('--out', type=str, required=True, help='Out directory')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--num_workers', type=int, required=True, help='Num workers')
    parser.add_argument('--epochs', type=int, required=True, help='Num epochs')
    parser.add_argument('--device', type=int, required=True, help='Gpu number')

    args = parser.parse_args()

    manifest_dir = args.manifest
    out_dir = args.out
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    device = args.device

    train_manifest_file = os.path.join(manifest_dir, 'nemo-train-manifest_processed.json')
    test_manifest_file = os.path.join(manifest_dir, 'nemo-test-manifest_processed.json')
    dev_manifest_file = os.path.join(manifest_dir, 'nemo-dev-manifest_processed.json')

    pretrained = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")

    train_manifest_data = read_manifest(train_manifest_file)
    train_charset = get_charset(train_manifest_data)

    dev_manifest_data = read_manifest(dev_manifest_file)
    dev_charset = get_charset(dev_manifest_data)

    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
    pretrained.change_vocabulary(new_vocabulary=list(train_dev_set))
    # pretrainedConfig = DictConfig(pretrained.cfg)
    pretrainedConfig = copy.deepcopy(pretrained.cfg)
    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    trainer = ptl.Trainer(gpus=[device],
                          accelerator=accelerator,
                          max_epochs=epochs,
                          accumulate_grad_batches=1,
                          enable_checkpointing=False,
                          logger=False,
                          log_every_n_steps=5,
                          check_val_every_n_epoch=10)
    # Setup train, validation, test configs
    with open_dict(pretrainedConfig):
        pretrainedConfig.train_ds.manifest_filepath = train_manifest_file
        pretrainedConfig.validation_ds.manifest_filepath = dev_manifest_file
        pretrainedConfig.test_ds.manifest_filepath = test_manifest_file
        pretrainedConfig['train_ds']['is_tarred'] = False
        pretrainedConfig['train_ds']['tarred_audio_filepaths'] = None
        pretrainedConfig['validation_ds']['is_tarred'] = False
        pretrainedConfig['validation_ds']['tarred_audio_filepaths'] = None
        pretrained.set_trainer(trainer)
        pretrained.setup_training_data(pretrainedConfig['train_ds'])
        pretrained.setup_validation_data(pretrainedConfig['validation_ds'])

    from nemo.utils import exp_manager
    from omegaconf import OmegaConf

    # Environment variable generally used for multi-node multi-gpu training.
    # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
    os.environ.pop('NEMO_EXPM_VERSION', None)

    exp_config = exp_manager.ExpManagerConfig(
        exp_dir=out_dir,
        name=f"Conformer-ctc",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
    )

    exp_config = OmegaConf.structured(exp_config)
    logdir = exp_manager.exp_manager(trainer, exp_config)

    # Train the model
    trainer.fit(pretrained)


if __name__ == '__main__':
    main()
