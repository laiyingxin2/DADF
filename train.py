import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist


# --------- global vars (keep names) ----------
local_rank = 0
device = None
rank = 0
world_size = 1


def ddp_setup(args):
    """
    Make train.py work for both torchrun and (legacy) torch.distributed.launch.

    torchrun: local rank should be read from env LOCAL_RANK.  :contentReference[oaicite:3]{index=3}
    """
    global local_rank, device, rank, world_size

    # init pg (env:// expects RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set by launcher)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # local rank: prefer env LOCAL_RANK (torchrun), fallback to args.local_rank (launch)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        if args.local_rank is None or args.local_rank < 0:
            raise RuntimeError(
                "Cannot determine local_rank. "
                "Please launch with torchrun or torch.distributed.launch."
            )
        local_rank = int(args.local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


def is_main_process():
    return rank == 0


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if is_main_process():
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    pbar = tqdm(total=len(loader), leave=False, desc='val') if is_main_process() else None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)

        inp = batch['inp']
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(world_size)]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(world_size)]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)

        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # NOTE: 这里原代码是 cat(dim=1)。我不确定你 pred/gt 的 batch 维度设计，
    # 先保持你的逻辑不改（如需我也可以帮你核对维度）。
    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)

    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)
    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1

    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))

    if is_main_process():
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    pbar = tqdm(total=len(train_loader), leave=False, desc='train') if is_main_process() else None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)

        inp = batch['inp']
        gt = batch['gt']

        model.set_input(inp, gt)
        model.optimize_parameters()

        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(world_size)]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)

    if is_main_process():
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    # ---- DDP wrap (DO NOT drop wrapper) ----
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    # ---- load SAM checkpoint ----
    ckpt = torch.load(config['sam_checkpoint'], map_location="cpu")
    # 兼容有些 ckpt 是 {"model": state_dict, ...}
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ddp_model.module.load_state_dict(state_dict, strict=False)

    # ---- freeze encoder (except prompt_generator) ----
    for name, para in ddp_model.module.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    if is_main_process():
        model_total_params = sum(p.numel() for p in ddp_model.module.parameters())
        model_grad_params = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()

        train_loss_G = train(train_loader, ddp_model.module)  # keep your model API
        lr_scheduler.step()

        if is_main_process():
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            save(config, ddp_model.module, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(
                val_loader, ddp_model.module, eval_type=config.get('eval_type')
            )

            if is_main_process():
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, ddp_model.module, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, ddp_model.module, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save(
                {"prompt": prompt_generator, "decode_head": decode_head},
                os.path.join(save_path, f"prompt_epoch_{name}.pth")
            )
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)

    # accept both --local_rank (launch) and --local-rank (torchrun older patterns)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=None, help="")

    args = parser.parse_args()

    # setup ddp first (so we can print only on rank0)
    ddp_setup(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if is_main_process():
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
