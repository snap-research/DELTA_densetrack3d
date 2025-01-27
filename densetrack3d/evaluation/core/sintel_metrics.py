import torch


def compute_sintel_metrics(pred, gt, valid=None):
    epe_all, px1, px3, px5 = get_epe(pred, gt, valid=valid)
    metrics = {
        "epe_all": epe_all.mean(),
        "px1": px1.mean(),
        "px3": px3.mean(),
        "px5": px5.mean(),
    }
    return metrics


def get_epe(pred, gt, valid=None):
    # breakpoint()
    # diff = torch.norm(pred - gt, p=2, dim=-1, keepdim=True)
    diff = torch.sum((pred - gt) ** 2, dim=-1).sqrt()

    if valid is not None:
        diff = diff[valid.bool()]

    px1 = (diff < 1.0).float().mean().cpu().numpy()
    px3 = (diff < 3.0).float().mean().cpu().numpy()
    px5 = (diff < 5.0).float().mean().cpu().numpy()

    epe_all = diff.mean().cpu().numpy()
    return epe_all, px1, px3, px5
