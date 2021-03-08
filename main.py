from argparse import ArgumentParser
import argparse
from os import name
import pytorch_lightning as pl 
from src.model.lightning_module.non_auto_regressive_transformer import NART
from pytorch_lightning.profiler import AdvancedProfiler, BaseProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from src.tools.data_module.hdf_module import DataModule
from warnings import filterwarnings
filterwarnings('ignore')


def main(args):
    dict_args = vars(args)
    model_name = 'NART'
    logger = TensorBoardLogger('ckpt', model_name)
    dir_path = f'{logger.save_dir}/{model_name}/version_{logger.version}/'
    file_name = '{epoch}-{val_loss: .4f}'
    model_checkpoint = ModelCheckpoint(
        dirpath=dir_path, filename=file_name, monitor='val_loss', verbose=True, save_top_k=5
    )
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = model_checkpoint,
        profiler=AdvancedProfiler('profile'),
        gradient_clip_val=5.0, gpus=[1], precision=16, amp_level='O1', amp_backend='native',
        reload_dataloaders_every_epoch=True, max_epochs=500, min_epochs=500,
        weights_summary=None,
        accumulate_grad_batches=4, 
        resume_from_checkpoint='ckpt/NART/version_31/epoch=12-val_loss= 2.1670.ckpt',
        flush_logs_every_n_steps=5000,
        log_every_n_steps=5000,
        limit_val_batches=0.5,
        )
    data = DataModule()
    model = NART(args)
    trainer.fit(model=model, datamodule=data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--model_name', type=str, default='NART')
    parser = NART.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)


    """
      logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, bool, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[Plugin, str, list]] = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        automatic_optimization: Optional[bool] = None,
        move_metrics_to_cpu: bool = False,
        enable_pl_optimizer: bool = None,  # todo: remove in v1.3
        multiple_trainloader_mode: str = 'max_size_cycle',
        stochastic_weight_avg: bool = False   
    
    
    """