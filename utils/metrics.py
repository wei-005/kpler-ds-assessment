def eval_topk_mrr(pred_lists, true_list, ks=(1,3,5)):
    hits = {k:0 for k in ks}
    mrr_sum = 0.0
    n = len(true_list)
    for preds, truth in zip(pred_lists, true_list):
        rank = None
        for i, p in enumerate(preds, start=1):
            if p == truth:
                rank = i; break
        if rank is not None:
            for k in ks:
                if rank <= k: hits[k] += 1
            mrr_sum += 1.0/rank
    res = {f"hits@{k}": (hits[k]/n if n else 0.0) for k in ks}
    res["mrr"] = (mrr_sum/n if n else 0.0)
    return res
