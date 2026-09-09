import os
import sys

# This entry point reuses the modules in the repository root; only the training
# and evaluation loop below is specific to node classification.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as t
from torch import nn
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from model import (Expert, Feat_Projector, Adj_Projector, AnyGraph,
                   CHECKPOINT_VERSION, describe_model, apply_checkpoint_config,
                   restore_trn_counts)
from data_handler import MultiDataHandler, NodeDataHandler
from dataset_groups import NODE_GROUPS, resolve_dataset_setting
import run_dir
import numpy as np
import pickle
import random
import os
import setproctitle
import time
from datetime import datetime, timezone
from sklearn.metrics import f1_score

def set_seed(seed):
    """Seed every RNG the training and evaluation paths draw from.

    Without this, negative sampling, the per-step dataset downsampling, the
    randomised SVD in the projectors, and graph-transformer anchor selection all
    vary between otherwise identical runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


class Exp:
    def __init__(self, multi_handler):
        self.multi_handler = multi_handler
        print(list(map(lambda x: x.data_name, multi_handler.trn_handlers)))
        for group_id, tst_handlers in enumerate(multi_handler.tst_handlers_group):
            print(f'Test group {group_id}', list(map(lambda x: x.data_name, tst_handlers)))
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss']
        tst_mets = ['Acc', 'F1', 'Loss', 'preLoss']
        mets = trn_mets + tst_mets
        for met in mets:
            if met in trn_mets:
                self.metrics['Train' + met] = list()
            if met in tst_mets:
                for i in range(len(self.multi_handler.tst_handlers_group)):
                    self.metrics['Test' + str(i) + met] = list()
        
    def make_print(self, name, ep, reses, save, data_name=None):
        if data_name is None:
            ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        else:
            ret = 'Epoch %d/%d, %s %s: ' % (ep, args.epoch, data_name, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric if data_name is None else name + data_name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '      '
        return ret
    
    def run(self):
        self.prepare_model()
        log('Model Prepared')
        stloc = 0
        if args.load_model != None:
            self.load_model()
            stloc = len(self.metrics['TrainLoss']) * args.tst_epoch - (args.tst_epoch - 1)
        best_acc, best_ep = 0, -1
        for ep in range(stloc, args.epoch):
            tst_flag = (ep % args.tst_epoch == 0)
            start_time = time.time()
            self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=True)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, tst_flag))
            self.multi_handler.remake_initial_projections()
            end_time = time.time()
            print(f'NOTICE: {end_time-start_time}')
            if tst_flag:
                group_accs = []
                for handler_group_id in range(len(self.multi_handler.tst_handlers_group)):
                    tst_handlers = self.multi_handler.tst_handlers_group[handler_group_id]
                    self.model.assign_experts(tst_handlers, reca=False, log_assignment=True)
                    acc, f1, tstnum = 0, 0, 0
                    for i, handler in enumerate(tst_handlers):
                        reses = self.test_epoch(handler, i)
                        log(self.make_print(handler.data_name, ep, reses, False))
                        acc += reses['Acc'] * reses['tstNum']
                        f1 += reses['F1'] * reses['tstNum']
                        tstnum += reses['tstNum']
                    reses = {'Acc': acc / tstnum, 'F1': f1 / tstnum}
                    log(self.make_print('Test'+str(handler_group_id), ep, reses, tst_flag))
                    group_accs.append(reses['Acc'])

                # Select on the last test group: the zero-shot group under the
                # 'A+B' setting, otherwise the only group.
                if group_accs[-1] > best_acc:
                    best_acc, best_ep = group_accs[-1], ep
                    self.save_model(run_dir.BEST_STEM)
                self.save_history()
            print()

        if best_ep >= 0:
            log('Best epoch %d: Acc = %.4f, saved as %s' % (
                best_ep, best_acc, os.path.join(args.run_path, run_dir.BEST_STEM + '.mod')))

        self.final_metrics = []
        for test_group_id in range(len(self.multi_handler.tst_handlers_group)):
            per_dataset = dict()
            repeat_times = 10
            overall_acc, overall_f1 = np.zeros(repeat_times), np.zeros(repeat_times)
            overall_tstnum = 0
            tst_handlers = self.multi_handler.tst_handlers_group[test_group_id]
            if args.assignment == 'one-graph-one-expert':
                self.model.assign_experts(tst_handlers, reca=False, log_assignment=True)
            for i, handler in enumerate(tst_handlers):
                    mets = dict()
                    for _ in range(repeat_times):
                        handler.make_projectors()
                        if not args.assignment == 'one-graph-one-expert':
                            self.model.assign_experts([handler], reca=False, log_assignment=False)
                        reses = self.test_epoch(handler, i if args.assignment == 'one-graph-one-expert' else 0)
                        log(self.make_print('Test', args.epoch, reses, False))
                        for met in reses:
                            if met not in mets:
                                mets[met] = []
                            mets[met].append(reses[met])
                    tstnum = reses['tstNum']
                    tot_reses = dict()
                    for met in reses:
                        tem_arr = np.array(mets[met])
                        tot_reses[met + '_std'] = tem_arr.std()
                        tot_reses[met + '_mean'] = tem_arr.mean()
                    
                    overall_acc += np.array(mets['Acc']) * tstnum
                    overall_f1 += np.array(mets['F1']) * tstnum
                    overall_tstnum += tstnum
                    per_dataset[handler.data_name] = tot_reses
                    log(self.make_print(f'Test', args.epoch, tot_reses, False, handler.data_name))
            overall_acc /= overall_tstnum
            overall_f1 /= overall_tstnum
            overall_res = dict()
            overall_res['Acc_mean'] = overall_acc.mean()
            overall_res['Acc_std'] = overall_acc.std()
            overall_res['F1_mean'] = overall_f1.mean()
            overall_res['F1_std'] = overall_f1.std()
            log(self.make_print('Overall Test', args.epoch, overall_res, False))
            self.final_metrics.append({
                'group': test_group_id,
                'datasets': [handler.data_name for handler in tst_handlers],
                'per_dataset': per_dataset,
                'overall': overall_res,
            })
        self.save_history()

    def print_model_size(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for param in self.model.parameters():
            tem = np.prod(param.size())
            total_params += tem
            if param.requires_grad:
                trainable_params += tem
            else:
                non_trainable_params += tem
        print(f'Total params: {total_params/1e6}')
        print(f'Trainable params: {trainable_params/1e6}')
        print(f'Non-trainable params: {non_trainable_params/1e6}')

    def prepare_model(self):
        self.model = AnyGraph()
        t.cuda.empty_cache()
        self.print_model_size()

    def train_epoch(self):
        self.model.train()
        trn_loader = self.multi_handler.joint_trn_loader
        trn_loader.dataset.neg_sampling()
        ep_loss, ep_preloss, ep_regloss = 0, 0, 0
        steps = len(trn_loader)
        tot_samp_num = 0
        counter = [0] * len(self.multi_handler.trn_handlers)
        reassign_steps = sum(list(map(lambda x: x.reproj_steps, self.multi_handler.trn_handlers)))
        for i, batch_data in enumerate(trn_loader):
            if args.epoch_max_step > 0 and i >= args.epoch_max_step:
                break
            ancs, poss, negs, dataset_id = batch_data
            ancs = ancs[0].long()
            poss = poss[0].long()
            negs = negs[0].long()
            dataset_id = dataset_id[0].long()
            tem_bar = self.multi_handler.trn_handlers[dataset_id].ratio_500_all
            if tem_bar < 1.0 and np.random.uniform() > tem_bar:
                steps -= 1
                continue

            expert = self.model.summon(dataset_id)#.cuda()
            opt = self.model.summon_opt(dataset_id)
            # adj = self.multi_handler.trn_handlers[dataset_id].trn_input_adj
            feats = self.multi_handler.trn_handlers[dataset_id].projectors
            loss, loss_dict = expert.cal_loss((ancs, poss, negs), feats)
            opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(expert.parameters(), max_norm=20, norm_type=2)
            opt.step()

            sample_num = ancs.shape[0]
            tot_samp_num += sample_num
            ep_loss += loss.item() * sample_num
            ep_preloss += loss_dict['preloss'].item() * sample_num
            ep_regloss += loss_dict['regloss'].item()
            log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

            counter[dataset_id] += 1
            if (counter[dataset_id] + 1) % self.multi_handler.trn_handlers[dataset_id].reproj_steps == 0:
            # if args.proj_trn_steps > 0 and counter[dataset_id] >= args.proj_trn_steps:
                self.multi_handler.trn_handlers[dataset_id].make_projectors()
            if (i + 1) % reassign_steps == 0:
                self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=False)
        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        t.cuda.empty_cache()
        return ret
    
    def make_trn_masks(self, numpy_usrs, csr_mat):
        trn_masks = csr_mat[numpy_usrs].tocoo()
        cand_size = trn_masks.shape[1]
        trn_masks = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long()
        return trn_masks, cand_size

    def test_loss_epoch(self, handler, dataset_id):
        with t.no_grad():
            tst_loader = handler.tst_loss_loader
            self.model.eval()
            expert = self.model.summon(dataset_id)#.cuda()
            ep_loss, ep_preloss, ep_regloss = 0, 0, 0
            steps = len(tst_loader)
            tot_samp_num = 0
            for i, batch_data in enumerate(tst_loader):
                ancs, poss, negs = batch_data
                ancs = ancs.long()
                poss = poss.long()
                negs = negs.long()
                # adj = handler.tst_input_adj
                feats = handler.projectors
                loss, loss_dict = expert.cal_loss((ancs, poss, negs), feats)
                
                sample_num = ancs.shape[0]
                tot_samp_num += sample_num
                ep_loss += loss.item() * sample_num
                ep_preloss += loss_dict['preloss'].item() * sample_num
                ep_regloss += loss_dict['regloss'].item()
                log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        ret['tot_samp_num'] = tot_samp_num
        t.cuda.empty_cache()
        return ret
    
    def test_epoch(self, handler, dataset_id):
        with t.no_grad():
            tst_loader = handler.tst_loader
            class_num = tst_loader.dataset.class_num
            self.model.eval()
            expert = self.model.summon(dataset_id)
            ep_acc, ep_tot = 0, 0
            steps = len(tst_loader)
            for i, batch_data in enumerate(tst_loader):
                nodes, labels = list(map(lambda x: x.long().cuda(), batch_data))
                feats = handler.projectors
                preds = expert.pred_for_node_test(nodes, class_num, feats, rerun_embed=False if i!=0 else True)
                if i == 0:
                    all_preds, all_labels = preds, labels
                else:
                    all_preds = t.concatenate([all_preds, preds])
                    all_labels = t.concatenate([all_labels, labels])
                hit = (labels == preds).float().sum().item()
                ep_acc += hit
                ep_tot += labels.shape[0]
                log('Steps %d/%d: hit = %d, tot = %d          ' % (i, steps, ep_acc, ep_tot), save=False, oneline=True)
        ret = dict()
        ret['Acc'] = ep_acc / ep_tot
        ret['F1'] = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        ret['tstNum'] = ep_tot
        t.cuda.empty_cache()
        return ret

    
    def calc_recall_ndcg(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
    
    def save_model(self, stem):
        content = {
            'version': CHECKPOINT_VERSION,
            'config': describe_model(self.model),
            'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }
        t.save(content, os.path.join(args.run_path, stem + '.mod'))
        log('Model Saved: %s' % stem)

    def save_history(self):
        if args.epoch == 0:
            return
        with open(os.path.join(args.run_path, run_dir.HISTORY_NAME), 'wb') as fs:
            pickle.dump(self.metrics, fs)
        self.save_model(run_dir.FINAL_STEM)

    def finalize(self, status):
        """Write the run's metrics and close out its manifest.

        Separate from save_history() because a test-only run (--epoch 0) writes
        no checkpoints but is still the experimental result worth keeping; those
        runs used to leave nothing behind at all.
        """
        run_dir.write_json(os.path.join(args.run_path, run_dir.METRICS_NAME),
                           getattr(self, 'final_metrics', []))
        fields = {'status': status,
                  'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        if getattr(self, 'model', None) is not None:
            fields['checkpoint'] = {'version': CHECKPOINT_VERSION,
                                    'config': describe_model(self.model)}
        run_dir.update_manifest(args.run_path, **fields)

    def rebuild_from_checkpoint(self, ckp):
        """Reconstruct the model from a checkpoint, in either format."""
        if 'state_dict' not in ckp:
            # Checkpoints published with the paper pickle the whole nn.Module,
            # so they only load while model.py's class names and module paths
            # are unchanged. Re-saving migrates them to the current format.
            log('Loaded a legacy whole-module checkpoint; re-save to migrate it')
            return ckp['model']
        apply_checkpoint_config(ckp['config'])
        model = AnyGraph()
        model.load_state_dict(ckp['state_dict'])
        restore_trn_counts(model, ckp['config'])
        return model

    def load_model(self):
        # weights_only=True (the torch>=2.6 default) cannot unpickle the legacy
        # whole-module checkpoints handled below.
        mod_path, his_path = run_dir.resolve_checkpoint(args.load_model)
        ckp = t.load(mod_path, map_location='cpu', weights_only=False)
        self.model = self.rebuild_from_checkpoint(ckp)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        # A missing history is not fatal: 'best' checkpoints never had one, so
        # the better model used to be the one that could not be loaded.
        if os.path.exists(his_path):
            with open(his_path, 'rb') as fs:
                self.metrics = pickle.load(fs)
        else:
            log('No history beside %s; starting metrics empty' % os.path.basename(mod_path))
        log('Model Loaded')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu.split(',')) == 2:
        args.devices = ['cuda:0', 'cuda:1']
    elif len(args.gpu.split(',')) > 2:
        raise Exception('Devices should be less than 2')
    else:
        args.devices = ['cuda:0', 'cuda:0']
    set_seed(args.seed)
    logger.saveDefault = True
    setproctitle.setproctitle('akaxia_AutoGraph')

    log('Start')


    trn_datasets, tst_datasets_group = resolve_dataset_setting(args.dataset_setting, NODE_GROUPS)

    run_id = run_dir.make_run_id(args, tst_datasets_group)
    args.run_path = run_dir.create_run_dir(run_id, args.overwrite)
    logger.set_logfile(os.path.join(args.run_path, run_dir.LOG_NAME))
    run_dir.start_manifest(args.run_path, run_id, args, trn_datasets, tst_datasets_group)
    log('Run %s' % args.run_path)

    handler = MultiDataHandler(trn_datasets, tst_datasets_group, handler_cls=NodeDataHandler)
    log('Load Data')

    exp = Exp(handler)
    try:
        exp.run()
    except BaseException:
        exp.finalize('failed')
        raise
    exp.finalize('completed')
    print(args.load_model, args.dataset_setting)
