import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class iFSLModule(pl.LightningModule):
    """
    """
    def __init__(self, args):
        super(iFSLModule, self).__init__()

        self.args = args
        self.way = self.args.way
        self.weak = args.weak
        self.range = torch.arange(args.way + 1, requires_grad=False).view(1, args.way + 1, 1, 1)
        self.learner = None
        self.test_total_episodes = 0
        self.test_labeled_episodes = 0

    def forward(self, batch):
        pass

    def train_mode(self):
        pass

    def configure_optimizers(self):
        pass

    def predict_mask_nshot(self, batch, nshot):
        pass

    def training_step(self, batch, batch_idx):
        """
        batch.keys()
        > dict_keys(['query_img', 'query_mask', 'query_name', 'query_ignore_idx', 'org_query_imsize', 'support_imgs', 'support_masks', 'support_names', 'support_ignore_idxs', 'class_id'])

        batch['query_img'].shape : [bsz, 3, H, W]
        batch['query_mask'].shape : [bsz, H, W]
        batch['query_name'].len : [bsz]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['org_query_imsize'].len : [bsz]
        batch['support_imgs'].shape : [bsz, way, shot, 3, H, W]
        batch['support_masks'].shape : [bsz, way, shot, H, W]
        # FYI: this support_names' shape is transposed so keep in mind for vis
        batch['support_names'].shape : [bsz, shot, way]
        batch['support_ignore_idxs'].shape: [bsz, way, shot, H, W]
        batch['class_id'].shape : [bsz]
        batch['support_classes'].shape : [bsz, way] (torch.int64)
        batch['query_class_presence'].shape : [bsz, way] (torch.bool)
        # FYI: K-shot is always fixed to 1 for training
        """

        split = 'trn' if self.training else 'val'
        shared_masks = self.forward(batch)
        pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)

        if self.weak:
            loss = self.compute_cls_objective(shared_masks, batch['query_class_presence'])
        else:
            loss = self.compute_seg_objective(logit_seg, batch['query_mask'])

        with torch.no_grad():
            self.average_meter.update_cls(pred_cls, batch['query_class_presence'])
            self.average_meter.update_seg(pred_seg, batch, loss.item())

            self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self._shared_epoch_end(training_step_outputs)

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self.training_step(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_epoch_end(validation_step_outputs)

    def _shared_epoch_end(self, steps_outputs):
        split = 'trn' if self.training else 'val'
        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        error_rate = self.average_meter.compute_cls_error_rate()
        loss = self.average_meter.avg_seg_loss()

        dict = {f'{split}/loss': loss,
                f'{split}/miou': miou,
            f'{split}/er': er,
            f'{split}/error_rate': error_rate}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, logger=True)

        space = '\n\n' if split == 'val' else '\n'
        print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er(exact): {er:.3f} | {split}/error_rate: {error_rate:.3f}')

    def _is_oem_sliding_enabled(self, batch):
        return False

    def _sliding_positions(self, length, tile, stride):
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] != length - tile:
            positions.append(length - tile)
        return positions

    def _normalize_raw_query(self, query_raw):
        query = query_raw.float().div(255.0).unsqueeze(0)
        mean = query.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = query.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (query - mean) / std

    def _build_single_sample_batch(self, batch, sample_idx):
        sample_batch = {}
        batch_size = batch['query_img'].shape[0]
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                sample_batch[key] = value[sample_idx:sample_idx + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                sample_batch[key] = [value[sample_idx]]
            else:
                sample_batch[key] = value
        return sample_batch

    def _predict_logit_mask_nshot(self, batch, nshot):
        if nshot <= 1:
            shared_masks = self.forward(batch)
            _, _, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
            return logit_seg

        logit_mask_agg = None
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()
        support_ignore_idxs = batch.get('support_ignore_idxs')
        if support_ignore_idxs is not None:
            support_ignore_idxs = support_ignore_idxs.clone()

        for s_idx in range(nshot):
            batch_s = dict(batch)
            batch_s['support_imgs'] = support_imgs[:, :, s_idx:s_idx + 1]
            batch_s['support_masks'] = support_masks[:, :, s_idx:s_idx + 1]
            if support_ignore_idxs is not None:
                batch_s['support_ignore_idxs'] = support_ignore_idxs[:, :, s_idx:s_idx + 1]

            shared_masks = self.forward(batch_s)
            _, _, logit_seg = self.predict_cls_and_mask(shared_masks, batch_s)
            if logit_mask_agg is None:
                logit_mask_agg = logit_seg
            else:
                logit_mask_agg += logit_seg

        return logit_mask_agg / float(nshot)

    def predict_mask_nshot_oem_sliding(self, batch, nshot):
        tile = max(1, int(getattr(self.args, 'oem_sw_tile', 400)))
        stride = max(1, int(getattr(self.args, 'oem_sw_stride', tile)))

        pred_cls_all = []
        pred_seg_all = []
        for sample_idx in range(batch['query_img'].shape[0]):
            sample_batch = self._build_single_sample_batch(batch, sample_idx)
            pred_cls_sample, pred_seg_sample = self.predict_mask_nshot(sample_batch, nshot)

            # Defensive: if this sample doesn't have a full raw query (e.g., tile-cropped episode),
            # or its raw query is smaller/equal than the tile, skip sliding aggregation and
            # use the single-shot prediction `pred_seg_sample` as-is.
            raw_q_all = batch.get('query_img_raw')
            raw_query = None
            if raw_q_all is None:
                raw_query = None
            elif torch.is_tensor(raw_q_all):
                raw_query = raw_q_all[sample_idx]
            elif isinstance(raw_q_all, (list, tuple)):
                raw_query = raw_q_all[sample_idx]

            if raw_query is None:
                pred_cls_all.append(pred_cls_sample.squeeze(0))
                pred_seg_all.append(pred_seg_sample.squeeze(0))
                continue

            query_h, query_w = int(raw_query.shape[-2]), int(raw_query.shape[-1])
            if query_h <= tile and query_w <= tile:
                pred_cls_all.append(pred_cls_sample.squeeze(0))
                pred_seg_all.append(pred_seg_sample.squeeze(0))
                continue

            ys = self._sliding_positions(query_h, tile, stride)
            xs = self._sliding_positions(query_w, tile, stride)

            logits_sum = None
            logits_count = torch.zeros((1, 1, query_h, query_w), device=sample_batch['query_img'].device)
            for y in ys:
                for x in xs:
                    query_tile_raw = raw_query[:, y:y + tile, x:x + tile]
                    query_tile = self._normalize_raw_query(query_tile_raw).to(sample_batch['query_img'].device)

                    batch_tile = dict(sample_batch)
                    batch_tile['query_img'] = query_tile
                    batch_tile['org_query_imsize'] = torch.tensor([query_tile_raw.shape[-1], query_tile_raw.shape[-2]],
                                                                  device=query_tile.device)

                    tile_logits = self._predict_logit_mask_nshot(batch_tile, nshot)
                    if logits_sum is None:
                        logits_sum = torch.zeros((1, tile_logits.shape[1], query_h, query_w), device=tile_logits.device)

                    tile_h, tile_w = tile_logits.shape[-2], tile_logits.shape[-1]
                    logits_sum[:, :, y:y + tile_h, x:x + tile_w] += tile_logits
                    logits_count[:, :, y:y + tile_h, x:x + tile_w] += 1.0

            logits_avg = logits_sum / torch.clamp(logits_count, min=1.0)
            pred_seg_sliding = logits_avg.argmax(dim=1)

            pred_cls_all.append(pred_cls_sample.squeeze(0))
            pred_seg_all.append(pred_seg_sliding.squeeze(0))

        pred_cls = torch.stack(pred_cls_all, dim=0)
        pred_seg = torch.stack(pred_seg_all, dim=0)
        return pred_cls, pred_seg

    def _filter_batch_by_mask(self, batch, valid_mask):
        filtered = {}
        batch_size = valid_mask.numel()
        valid_mask_cpu = valid_mask.detach().cpu().tolist()
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                filtered[key] = value[valid_mask]
            elif isinstance(value, list) and len(value) == batch_size:
                filtered[key] = [elem for elem, keep in zip(value, valid_mask_cpu) if keep]
            else:
                filtered[key] = value
        return filtered

    def test_step(self, batch, batch_idx):
        if self._is_oem_sliding_enabled(batch):
            pred_cls, pred_seg = self.predict_mask_nshot_oem_sliding(batch, self.args.shot)
        else:
            pred_cls, pred_seg = self.predict_mask_nshot(batch, self.args.shot)

        er_b = None
        iou_b = None
        has_query_mask = batch.get('has_query_mask')
        if has_query_mask is None:
            valid_mask = torch.ones(pred_seg.shape[0], dtype=torch.bool, device=pred_seg.device)
        elif torch.is_tensor(has_query_mask):
            valid_mask = has_query_mask.to(pred_seg.device).bool().view(-1)
        else:
            valid_mask = torch.ones(pred_seg.shape[0], dtype=torch.bool, device=pred_seg.device)

        if valid_mask.any():
            batch_eval = self._filter_batch_by_mask(batch, valid_mask)
            pred_cls_eval = pred_cls[valid_mask]
            pred_seg_eval = pred_seg[valid_mask]

            er_b = self.average_meter.update_cls(pred_cls_eval, batch_eval['query_class_presence'], loss=None)
            iou_b = self.average_meter.update_seg(pred_seg_eval, batch_eval, loss=None)

            self.average_meter.update_f1_metrics(pred_seg_eval.cpu(),
                                                 batch_eval['query_mask'].to(torch.int).cpu(),
                                                 batch_eval.get('query_ignore_idx'))

        self.test_total_episodes += int(valid_mask.numel())
        self.test_labeled_episodes += int(valid_mask.sum().item())

        if self.args.vis:
            print(batch_idx, 'qry:', batch['query_name'])
            print(batch_idx, 'spt:', batch['support_names'])
            if self.args.shot > 1: raise NotImplementedError
            if self.args.weak:
                batch['support_masks'] = torch.zeros(1, self.way, 400, 400).cuda()
            from common.vis import Visualizer
            Visualizer.initialize(True, self.way)
            Visualizer.visualize_prediction_batch(batch['support_imgs'].squeeze(2),
                                                  batch['support_masks'].squeeze(2),
                                                  batch['query_img'],
                                                  batch['query_mask'],
                                                  batch['org_query_imsize'],
                                                  pred_seg,
                                                  batch_idx,
                                                  iou_b=iou_b,
                                                  er_b=er_b,
                                                  to_cpu=True)

    def test_epoch_end(self, test_step_outputs):
        def to_float(value):
            if torch.is_tensor(value):
                return float(value.item())
            return float(value)

        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        error_rate = self.average_meter.compute_cls_error_rate()

        precisions, recalls, f1_scores = self.average_meter.calculate_f1_metrics()

        if self.test_labeled_episodes == 0:
            print('[test] No labeled query masks found in this split. Reporting metrics as NaN.')
            miou = float('nan')
            er = float('nan')
            error_rate = float('nan')
            precisions = [float('nan') for _ in precisions]
            recalls = [float('nan') for _ in recalls]
            f1_scores = [float('nan') for _ in f1_scores]

        length = 16
        dict = {'benchmark'.ljust(length): self.args.benchmark,
                'fold'.ljust(length): self.args.fold,
                'test/miou'.ljust(length): to_float(miou),
                'test/er'.ljust(length): to_float(er),
                'test/error_rate'.ljust(length): to_float(error_rate),
                'test/Precisions'.ljust(length): precisions,
                'test/Recalls'.ljust(length): recalls,
                'test/F1-scores'.ljust(length): f1_scores}

        for k in dict:
            self.log(k, dict[k], on_epoch=True)

        self.test_total_episodes = 0
        self.test_labeled_episodes = 0

    def predict_cls_and_mask(self, shared_masks, batch):
        logit_seg = self.merge_bg_masks(shared_masks)
        logit_seg = self.upsample_logit_mask(logit_seg, batch)

        with torch.no_grad():
            pred_cls = self.collect_class_presence(shared_masks)
            pred_seg = logit_seg.argmax(dim=1)

        return pred_cls, pred_seg, logit_seg

    def collect_class_presence(self, logit_mask):
        ''' logit_mask: B, N, 2, H, W '''
        # since logit_mask is log-softmax-ed, we use torch.log(0.5) for the threshold
        class_activation = logit_mask[:, :, 1].max(dim=-1)[0].max(dim=-1)[0] >= math.log(0.5)
        return class_activation.type(logit_mask.dtype).detach()

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_seg_objective(self, logit_mask, gt_mask):
        ''' supports 1-way training '''
        return F.nll_loss(logit_mask, gt_mask.long())

    def compute_cls_objective(self, shared_masks, gt_presence):
        ''' supports 1-way training '''
        # B, N, 2, H, W -> B, N, 2 -> B, 2
        prob_avg = shared_masks.mean(dim=[-1, -2]).squeeze(1)
        return F.nll_loss(prob_avg, gt_presence.long().squeeze(-1))

    def merge_bg_masks(self, shared_fg_masks):
        # B, N, H, W
        logit_fg = shared_fg_masks[:, :, 1]
        # B, 1, H, W
        logit_episodic_bg = shared_fg_masks[:, :, 0].mean(dim=1)
        # B, (1 + N), H, W
        logit_mask = torch.cat((logit_episodic_bg.unsqueeze(1), logit_fg), dim=1)

        return logit_mask

    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
