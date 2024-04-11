"""
U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation
Authors: Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich
"""

import lightning as L

import config
from models.mc_dropout import MC_Dropout
from datasets.cityscapes import CityscapesDataModule
from datasets.acdc import ACDCDataModule


if __name__ == '__main__':
    model = MC_Dropout()

    if config.DATASET == 'Cityscapes':
        data_module = CityscapesDataModule(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )
    
    elif config.DATASET == 'ACDC':
        data_module = ACDCDataModule(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator='gpu',
        strategy='ddp',
        devices=config.DEVICES,
        precision=config.PRECISION,
        check_val_every_n_epoch=1,
        # logger=WandbLogger(entity=config.ENTITY, project=config.PROJECT, name=config.RUN_NAME, save_dir='./logs', log_model=True),
        # callbacks=[
            # ModelCheckpoint(dirpath=f'./checkpoints/{config.RUN_NAME}', filename='model-{epoch}-{val_iou:.3f}', monitor='val_iou', mode='max', save_top_k=3),
            # LearningRateMonitor(logging_interval='epoch'),
        # ],
    )

    trainer.fit(model, data_module)