import torch as t 
import pytorch_lightning as pl
from model.module.decoder import DecoderOutput
from src.model.module import Encoder
from src.model.module import Decoder
from src.model.module import TriggerDotAttentionMask
from src.model.module import InputRegularizer, MaskRegularizer
from src.model.metrics import LabelSmoothingLoss
from src.tools.ranger import Ranger
from argparse import ArgumentParser
from src.model.layer.spec_augment import SpecAugment


class NART(pl.LightningModule):
    def __init__(self, hparams):
        super(NART, self).__init__()
        self.hparams = hparams
        self.lr = hparams.lr
        self.spec_augment = SpecAugment(
            n_time_mask=hparams.n_time_mask, 
            n_freq_mask=hparams.n_freq_mask,
            time_mask_length=hparams.time_mask_length, 
            freq_mask_length=hparams.freq_mask_length,
            p=0.2
        )
        self.encoder = Encoder(
            model_size=hparams.model_size, 
            dropout=hparams.dropout,
            feed_forward_size=hparams.feed_forward_size,
            hidden_size=hparams.hidden_size,
            num_layer=hparams.encoder_num_layer,
            left=hparams.left, 
            right=hparams.right,
            num_head=hparams.num_head,
            vocab_size=hparams.vocab_size
        )
        self.trigger = t.jit.script(
            TriggerDotAttentionMask(blank_id=hparams.blank_id, trigger_eps=6)
        )
        self.ipr = InputRegularizer()
        self.mr = MaskRegularizer()
        self.decoder = Decoder(
            model_size=hparams.model_size, 
            dropout=hparams.dropout,
            feed_forward_size=hparams.feed_forward_size,
            hidden_size=hparams.hidden_size,
            num_layer=hparams.decoder_num_layer, 
            num_head=hparams.num_head, 
            vocab_size=hparams.vocab_size, 
            blank_id=hparams.blank_id, 
            place_id=hparams.place_id, 
            max_length=hparams.decoder_max_length
        )
        self.att_ce_loss = LabelSmoothingLoss(hparams.vocab_size)
        self.att_lg_loss = LabelSmoothingLoss(4)

    def decode(self, feature, feature_length):
        encoded_feature, ctc_language_output, ctc_output, feature_length, feature_max_length = self.encoder(
            feature, feature_length
        )
        input_, input_mask, trigger_mask = self.trigger(
            ctc_output
        )
        decoder_output, decoder_language_output = self.decoder(
            input_, input_mask, encoded_feature, trigger_mask
        )
        output_id = t.argmax(decoder_output, -1)
        return output_id
    
    def training_step(self, batch, batch_idx):
        feature, feature_length, ctc_target, att_input, att_output, target_length, ctc_lan_target, att_lan_target = \
            batch[0], batch[1],batch[2], batch[3], batch[4],batch[5],batch[6],batch[7]
        feature = self.spec_augment(feature, feature_length)
        encoded_feature, ctc_language_output, ctc_output, feature_length, feature_max_length = self.encoder(feature, feature_length)
        input_, input_mask, trigger_mask = self.trigger(ctc_output)
        input_, input_mask = self.ipr(input_, input_mask, att_output.size(1))
        trigger_mask = self.mr(trigger_mask, att_output.size(1))
        decoder_output, decoder_language_output = self.decoder(input_, input_mask, encoded_feature, trigger_mask)

        loss, ctc_loss, att_loss, ctc_lan_loss, att_lan_loss = self.cal_loss(
            feature_length, ctc_target, att_output, target_length,
            ctc_lan_target, att_lan_target, ctc_output, ctc_language_output, decoder_output,
            decoder_language_output
        )
        log_dict = {'train_loss': loss.item(), 'att_loss': att_loss.item(), 'ctc_loss': ctc_loss.item(), 'att_lan_loss': att_lan_loss.item(), 'ctc_lan_loss': ctc_lan_loss.item()}
        self.log_dict(dictionary=log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss, 'att_loss': att_loss, 'ctc_loss': ctc_loss, 'att_lan_loss': att_lan_loss, 'ctc_lan_loss': ctc_lan_loss}

    def validation_step(self, batch, batch_idx):
        feature, feature_length, ctc_target, att_input, att_output, target_length, ctc_lan_target, att_lan_target = \
            batch[0], batch[1],batch[2], batch[3], batch[4],batch[5],batch[6],batch[7]
        encoded_feature, ctc_language_output, ctc_output, feature_length, feature_max_length = self.encoder(feature, feature_length)
        input_, input_mask, trigger_mask = self.trigger(ctc_output)
        input_, input_mask = self.ipr(input_, input_mask, att_output.size(1))
        trigger_mask = self.mr(trigger_mask, att_output.size(1))
        decoder_output, decoder_language_output = self.decoder(input_, input_mask, encoded_feature, trigger_mask)

        loss, ctc_loss, att_loss, ctc_lan_loss, att_lan_loss = self.cal_loss(
            feature_length, ctc_target, att_output, target_length,
            ctc_lan_target, att_lan_target, ctc_output, ctc_language_output, decoder_output,
            decoder_language_output
        )
        log_dict = {'val_loss': loss.item(), 'val_att_loss': att_loss.item(), 'val_ctc_loss': ctc_loss.item(), 'val_att_lan_loss': att_lan_loss.item(), 'val_ctc_lan_loss': ctc_lan_loss.item()}
        self.log_dict(dictionary=log_dict, prog_bar=False, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_att_loss': att_loss, 'val_ctc_loss': ctc_loss, 'val_att_lan_loss': att_lan_loss, 'val_ctc_lan_loss': ctc_lan_loss}

    def test_step(self, batch, batch_idx):
        feature, feature_length, ctc_target, att_input, att_output, target_length, ctc_lan_target, att_lan_target = \
            batch[0], batch[1],batch[2], batch[3], batch[4],batch[5],batch[6],batch[7]
        encoded_feature, ctc_language_output, ctc_output, feature_length, feature_max_length = self.encoder(feature, feature_length)
        input_, input_mask, trigger_mask = self.trigger(ctc_output)
        decoder_output, decoder_language_output = self.decoder(input_, input_mask, encoded_feature, trigger_mask)
        decoder_output_id = t.argmax(decoder_output, -1)
        return decoder_output_id

    def validation_epoch_end(self, outputs):
        val_loss = t.stack([i['val_loss'] for i in outputs]).mean()
        att_loss = t.stack([i['val_att_loss'] for i in outputs]).mean()
        ctc_loss = t.stack([i['val_ctc_loss'] for i in outputs]).mean()
        att_lan_loss = t.stack([i['val_att_lan_loss'] for i in outputs]).mean()
        ctc_lan_loss = t.stack([i['val_ctc_lan_loss'] for i in outputs]).mean()
        log_dict = {'val_loss': val_loss.item(), 'val_att_loss': att_loss.item(), 'val_ctc_loss': ctc_loss.item(), 'val_att_lan_loss': att_lan_loss.item(), 'val_ctc_lan_loss': ctc_lan_loss.item()}
        self.log_dict(dictionary=log_dict, prog_bar=False, on_step=False, on_epoch=True)
        print(log_dict)
        ret = {
            'val_loss': val_loss, 'val_ctc_loss': ctc_loss, 'val_att_loss': att_loss, 'val_ctc_lan_loss': ctc_lan_loss, 'val_att_lan_loss': att_lan_loss,
            'log': {'val_loss': val_loss, 'val_ctc_loss': ctc_loss, 'val_att_loss': att_loss, 'val_ctc_lan_loss': ctc_lan_loss, 'val_att_lan_loss': att_lan_loss}
            }
        return ret

    def configure_optimizers(self):
        opt = Ranger(self.parameters(), lr = self.lr)
        return opt

    def cal_loss(
        self, feature_length, ctc_target, att_output, target_length,
        ctc_lan_target, att_lan_target, ctc_output, ctc_language_output, decoder_output,
        decoder_language_output):
        ctc_loss = self._cal_ctc_loss(ctc_output, ctc_target, feature_length, target_length)
        att_loss = self._cal_att_loss(decoder_output, att_output)
        ctc_lan_loss = self._cal_ctc_language_loss(ctc_language_output, ctc_lan_target, feature_length, target_length)
        att_lan_loss = self._cal_att_language_loss(decoder_language_output, att_lan_target)
        # att_loss, ctc_lan_loss, att_lan_loss = 0, 0, 0
        loss = ctc_loss * self.hparams.ctc_weight + att_loss * (1-0.1-self.hparams.ctc_weight) \
            + ctc_lan_loss * 0.05 + att_lan_loss * 0.05
        return loss, ctc_loss, att_loss, ctc_lan_loss, att_lan_loss

    def _cal_ctc_loss(self, ctc_output, ctc_target, feature_length, target_length):
        prob = t.nn.functional.log_softmax(ctc_output, -1)
        ctc_loss = t.nn.functional.ctc_loss(
            prob.transpose(0, 1), ctc_target, feature_length, target_length,
            blank=5, zero_infinity=True
        )
        return ctc_loss

    def _cal_ctc_language_loss(self,ctc_language_output, ctc_lan_target, feature_length, target_length):
        prob = t.nn.functional.log_softmax(ctc_language_output, -1)
        ctc_loss = t.nn.functional.ctc_loss(
            prob.transpose(0, 1), ctc_lan_target, feature_length, target_length,
            blank=4, zero_infinity=True
        )
        return ctc_loss

    def _cal_att_loss(self, decoder_output, att_output):
        return self.att_ce_loss(decoder_output, att_output)

    def _cal_att_language_loss(self,decoder_language_output, att_lan_target):
        return self.att_lg_loss(decoder_language_output, att_lan_target)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_size', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--feed_forward_size', type=int, default=1024)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--encoder_num_layer', type=int, default=12)
        parser.add_argument('--left', type=int, default=250)
        parser.add_argument('--right', type=int, default=3)
        parser.add_argument('--num_head', type=int, default=4)
        parser.add_argument('--vocab_size', type=int, default=6000)
        parser.add_argument('--blank_id', type=int, default=5)
        parser.add_argument('--trigger_eps', type=int, default=6)
        parser.add_argument('--decoder_num_layer', type=int, default=6)
        parser.add_argument('--place_id', type=int, default=9)
        parser.add_argument('--place_en_id', type=int, default=10)
        parser.add_argument('--decoder_max_length', type=int, default=150)
        parser.add_argument('--n_time_mask', type=int, default=2)
        parser.add_argument('--n_freq_mask', type=int, default=1)
        parser.add_argument('--time_mask_length', type=int, default=80)
        parser.add_argument('--freq_mask_length', type=int, default=20)
        parser.add_argument('--ctc_weight', type=float, default=0.4)
        return parser
