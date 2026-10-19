def build_student_optimizer(model, config):
    if (config.optimizer, config.scheduler, config.loss) != ("AdamW", "cosine", "huber"):
        raise ValueError("student setup expects AdamW, cosine scheduling, and Huber loss")
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    criterion = torch.nn.HuberLoss(delta=config.huber_delta)
    return optimizer, scheduler, criterion
