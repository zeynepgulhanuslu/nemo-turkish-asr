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

    train_manifest_file = os.path.join(manifest_dir, 'nemo-train-manifest.json')
    test_manifest_file = os.path.join(manifest_dir, 'nemo-test-manifest.json')
    dev_manifest_file = os.path.join(manifest_dir, 'nemo-dev-manifest.json')

    train_manifest = read_manifest(train_manifest_file)
    test_manifest = read_manifest(test_manifest_file)
    dev_manifest = read_manifest(dev_manifest_file)

    train_text = [data['text'] for data in train_manifest]
    dev_text = [data['text'] for data in dev_manifest]
    test_text = [data['text'] for data in test_manifest]

    train_charset = get_charset(train_manifest)
    dev_charset = get_charset(dev_manifest)
    test_charset = get_charset(test_manifest)

    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
    test_set = set(test_charset.keys())

    print(f"Number of tokens in train+dev set : {len(train_dev_set)}")
    print(f"Number of tokens in test set : {len(test_set)}")

    # OOV tokens in test set
    train_test_common = set.intersection(train_dev_set, test_set)
    test_oov = test_set - train_test_common
    print(f"Number of OOV tokens in test set : {len(test_oov)}")
    print()
    print(test_oov)

    # Populate dictionary mapping count: list[tokens]
    train_counts = defaultdict(list)
    for token, count in train_charset.items():
        train_counts[count].append(token)
    for token, count in dev_charset.items():
        train_counts[count].append(token)

    # Compute sorter order of the count keys
    count_keys = sorted(list(train_counts.keys()))

    # List of pre-processing functions
    PREPROCESSORS = [
        remove_special_characters]

    # Apply preprocessing
    train_data_processed = apply_preprocessors(train_manifest, PREPROCESSORS)
    dev_data_processed = apply_preprocessors(dev_manifest, PREPROCESSORS)
    test_data_processed = apply_preprocessors(test_manifest, PREPROCESSORS)

    # Write new manifests
    train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest_file)
    dev_manifest_cleaned = write_processed_manifest(dev_data_processed, dev_manifest_file)
    test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest_file)

    train_manifest_data = read_manifest(train_manifest_cleaned)
    train_charset = get_charset(train_manifest_data)

    dev_manifest_data = read_manifest(dev_manifest_cleaned)
    dev_charset = get_charset(dev_manifest_data)

    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
    print(f"Number of tokens in preprocessed train+dev set : {len(train_dev_set)}")

    char_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_conformer_ctc_small")
    char_model.change_vocabulary(new_vocabulary=list(train_dev_set))

    # @title Freeze Encoder { display-mode: "form" }
    freeze_encoder = True  # @param ["False", "True"] {type:"raw"}
    freeze_encoder = bool(freeze_encoder)

    def enable_bn_se(m):
        if type(m) == nn.BatchNorm1d:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

        if 'SqueezeExcite' in type(m).__name__:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

    if freeze_encoder:
        char_model.encoder.freeze()
        char_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        char_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")

    char_model.cfg.labels = list(train_dev_set)

    cfg = copy.deepcopy(char_model.cfg)

    # Setup train, validation, test configs
    print(cfg.train_ds)
    # Setup train, validation, test configs
    with open_dict(cfg):
        # Train dataset  (Concatenate train manifest cleaned and dev manifest cleaned)
        cfg.train_ds.manifest_filepath = f"{train_manifest_cleaned},{dev_manifest_cleaned}"
        cfg.train_ds.labels = list(train_dev_set)
        cfg.train_ds.normalize_transcripts = False
        cfg.train_ds.batch_size = batch_size
        cfg.train_ds.num_workers = num_workers
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True

        # Validation dataset  (Use test dataset as validation, since we train using train + dev)
        cfg.validation_ds.manifest_filepath = test_manifest_cleaned
        cfg.validation_ds.labels = list(train_dev_set)
        cfg.validation_ds.normalize_transcripts = False
        cfg.validation_ds.batch_size = batch_size
        cfg.validation_ds.num_workers = num_workers
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True

    # setup data loaders with new configs
    char_model.setup_training_data(cfg.train_ds)
    char_model.setup_multiple_validation_data(cfg.validation_ds)

    # Original optimizer + scheduler
    print(OmegaConf.to_yaml(char_model.cfg.optim))

    with open_dict(char_model.cfg.optim):
        char_model.cfg.optim.lr = 0.01
        char_model.cfg.optim.betas = [0.95, 0.5]  # from paper
        char_model.cfg.optim.weight_decay = 0.001  # Original weight decay
        char_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
        char_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
        char_model.cfg.optim.sched.min_lr = 1e-5

    # @title Metric
    use_cer = True  # @param ["False", "True"] {type:"raw"}
    log_prediction = True  # @param ["False", "True"] {type:"raw"}

    char_model._wer.use_cer = use_cer
    char_model._wer.log_prediction = log_prediction

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    print("accelerator :", accelerator)
    # # specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
    # Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True)

    trainer = ptl.Trainer(gpus=[device],
                          accelerator=accelerator,
                          max_epochs=epochs,
                          accumulate_grad_batches=1,
                          enable_checkpointing=False,
                          logger=False,
                          log_every_n_steps=5,
                          check_val_every_n_epoch=10)

    # Setup model with the trainer
    char_model.set_trainer(trainer)
    print("trainer set")
    # Finally, update the model's internal config
    char_model.cfg = char_model._cfg

    # Environment variable generally used for multi-node multi-gpu training.
    # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
    os.environ.pop('NEMO_EXPM_VERSION', None)

    config = exp_manager.ExpManagerConfig(
        exp_dir=out_dir,
        name="ASR-Char-Model-Language-tr",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
    )
    print("model configuration completed")

    config = OmegaConf.structured(config)
    print('labels:', char_model.cfg.labels)
    logdir = exp_manager.exp_manager(trainer, config)
    print('training starts here')
    trainer.fit(char_model)
    '''
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_medium")

    from omegaconf import OmegaConf
    params = OmegaConf.load("conf/conformer_ctc_char.yaml")
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
