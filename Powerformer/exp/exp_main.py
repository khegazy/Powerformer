from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, Powerformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            #'PatchTST': PatchTST,
            'Powerformer': Powerformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            print("MULTIGPU", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            #model = nn.DataParallel(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                            #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                        outputs = self.model(batch_x)
                        #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print("GOT DATA")
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                            #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                            #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, save_attn=False, output_dir="./", save_setting=None):
        test_data, test_loader = self._get_data(flag='test')
        if save_setting == None:
            save_setting = setting
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        folder_path = os.path.join(output_dir, 'results', save_setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if os.path.exists(os.path.join(folder_path, "score_bins.npy")):
            print(f"Found results for {setting} now exiting.")
            return


        preds = []
        trues = []
        inputx = []
        attn_raw_scores = [[]]
        attn_powerlaw_scores = [[]]
        attn_raw_weights = [[]]
        attn_powerlaw_weights = [[]]
        if self.args.model.lower() == 'transformer':
            for i in range(2):
                attn_raw_scores.append([])
                attn_powerlaw_scores.append([])
                attn_raw_weights.append([])
                attn_powerlaw_weights.append([])
        """
        folder_path = output_dir + 'test_results/' + save_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        """

        self.model.eval()
        if save_attn:
            if self.args.model.lower() == 'transformer':
                for enc in self.model.encoder.attn_layers:
                    enc.attention.inner_attention.record_scores = True
                for dec in self.model.decoder.layers:
                    dec.self_attention.inner_attention.record_scores = True
                for dec in self.model.decoder.layers:
                    dec.cross_attention.inner_attention.record_scores = True
            else:
                if self.model.decomposition:
                    for enc in self.model.model_trend.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True
                    for enc in self.model.model_res.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True
                else:
                    for enc in self.model.model.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True
        with torch.no_grad():
            score_bins = np.linspace(-500, 500, 2001)
            weight_bins = np.linspace(0, 1, 101)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                            #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                            #outputs = self.model.evaluate(batch_x, self.args.pred_len)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if save_attn:
                    if self.args.model.lower() == 'transformer':
                        raw_scores = []
                        powerlaw_scores = []
                        raw_weights = []
                        powerlaw_weights = []
                        attn_layers = [
                            self.model.encoder.attn_layers,
                            self.model.decoder.layers,
                            self.model.decoder.layers
                        ]
                        for idx, layers in enumerate(attn_layers):
                            raw_scores.append([])
                            powerlaw_scores.append([])
                            raw_weights.append([])
                            powerlaw_weights.append([])
                            for layer in layers:
                                if idx == 0:
                                    attn = layer.attention
                                elif idx == 1:
                                    attn = layer.self_attention
                                else:
                                    attn = layer.cross_attention
                                print("Raw weights", attn.inner_attention.raw_weights.detach().cpu().numpy().shape)
                                raw_scores[-1].append(
                                    np.histogram(
                                        attn.inner_attention.raw_scores.detach().cpu().numpy().flatten(),
                                        score_bins)[0]
                                )
                                powerlaw_scores[-1].append(
                                    np.histogram(
                                        attn.inner_attention.masked_scores.detach().cpu().numpy().flatten(),
                                        score_bins)[0]
                                )
                                raw_weights[-1].append(
                                    np.histogram(
                                        attn.inner_attention.raw_weights.detach().cpu().numpy().flatten(),
                                        weight_bins)[0]
                                )
                                powerlaw_weights[-1].append(
                                    np.histogram(
                                        attn.inner_attention.attn_weights.detach().cpu().numpy().flatten(),
                                        weight_bins)[0]
                                )
                    else:
                        raw_scores = [[
                            np.histogram(
                                enc.self_attn.sdp_attn.raw_scores.detach().cpu().numpy().flatten(),
                                score_bins)[0]\
                            for enc in self.model.model.backbone.encoder.layers
                        ]]
                        powerlaw_scores = [[
                            np.histogram(
                                enc.self_attn.sdp_attn.masked_scores.detach().cpu().numpy().flatten(),
                                score_bins)[0]\
                            for enc in self.model.model.backbone.encoder.layers
                        ]]
                        raw_weights = [[
                            np.histogram(
                                enc.self_attn.sdp_attn.raw_weights.detach().cpu().numpy().flatten(),
                                weight_bins)[0]\
                            for enc in self.model.model.backbone.encoder.layers
                        ]]
                        powerlaw_weights = [[
                            np.histogram(
                                enc.self_attn.sdp_attn.attn_weights.detach().cpu().numpy().flatten(),
                                weight_bins)[0]\
                            for enc in self.model.model.backbone.encoder.layers
                        ]]

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if save_attn:
                    print("combine raw", len(raw_scores), np.array(raw_weights[0]).shape)
                    for i in range(len(raw_scores)):
                        attn_raw_scores[i].append(np.array(raw_scores[i]))
                        attn_powerlaw_scores[i].append(np.array(powerlaw_scores[i]))
                        attn_raw_weights[i].append(np.array(raw_weights[i]))
                        attn_powerlaw_weights[i].append(np.array(powerlaw_weights[i]))
                """
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                """

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        if not save_attn:
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("result.txt", 'a')
            f.write(save_setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()

        pred_mse = (np.sum(preds - trues, axis=1))**2
        pred_mae = np.abs(np.sum(preds - trues, axis=1))
        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'mse.npy'), pred_mse)
        np.save(os.path.join(folder_path, 'mae.npy'), pred_mae)
        del preds, pred_mse, pred_mae
        idx0 = setting.find("_")
        idx1 = setting.find("_pl")
        idx2 = setting.find("_", idx1+2)
        data_filename = os.path.join(
            output_dir, "results",
            f"data_{setting[:idx0]}{setting[idx1:idx2]}.npy"
        )
        np.save(data_filename, trues)
        if save_attn:
            labels = ["encoder_SA_", "decoder_SA_", "decoder_CA_"]
            np.save(os.path.join(folder_path, 'score_bins.npy'), score_bins)
            np.save(os.path.join(folder_path, 'weight_bins.npy'), weight_bins)
            print("Saving", len(raw_scores), np.array(raw_scores[0]).shape)
            for idx in range(len(raw_scores)):
                label = labels[idx]
                print("LABEL", label)
                comb_attn_raw_scores = np.sum(np.array(attn_raw_scores[idx]), 0)
                comb_attn_powerlaw_scores = np.sum(np.array(attn_powerlaw_scores[idx]), 0)
                comb_attn_raw_weights = np.sum(np.array(attn_raw_weights[idx]), 0)
                comb_attn_powerlaw_weights = np.sum(np.array(attn_powerlaw_weights[idx]), 0)
                print("SIZES", np.array(attn_raw_scores[idx]).shape, len(attn_raw_scores), comb_attn_raw_scores.shape)
                np.save(os.path.join(folder_path, label + 'attn_raw_scores.npy'), comb_attn_raw_scores)
                np.save(os.path.join(folder_path, label + 'attn_powerlaw_scores.npy'), comb_attn_powerlaw_scores)
                np.save(os.path.join(folder_path, label + 'attn_raw_weights.npy'), comb_attn_raw_weights)
                np.save(os.path.join(folder_path, label + 'attn_powerlaw_weights.npy'), comb_attn_powerlaw_weights)
                
                if self.args.model.lower() == 'transformer':
                    if idx == 0:
                       decay_mask = self.model.encoder.attn_layers[0].attention.inner_attention.powerlaw_mask.detach().cpu().numpy() 
                    if idx == 1:
                       decay_mask = self.model.decoder.layers[0].self_attention.inner_attention.powerlaw_mask.detach().cpu().numpy() 
                    if idx == 2:
                       decay_mask = self.model.decoder.layers[0].cross_attention.inner_attention.powerlaw_mask.detach().cpu().numpy() 
                else:
                    decay_mask = self.model.model.backbone.encoder.layers[0].self_attn.sdp_attn.powerlaw_mask.detach().cpu().numpy()
                np.save(os.path.join(folder_path, label + 'powerlaw_mask.npy'), decay_mask)
            if self.args.model.lower() == 'transformer':
                for enc in self.model.encoder.attn_layers:
                    enc.attention.inner_attention.record_scores = True
                for dec in self.model.decoder.layers:
                    dec.self_attention.inner_attention.record_scores = True
                for dec in self.model.decoder.layers:
                    dec.cross_attention.inner_attention.record_scores = True
            else:
                if self.model.decomposition:
                    for enc in self.model.model_trend.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True
                    for enc in self.model.model_res.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True
                else:
                    for enc in self.model.model.backbone.encoder.layers:
                        enc.self_attn.sdp_attn.record_scores = True

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'ower' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
