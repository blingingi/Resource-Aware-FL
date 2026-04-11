#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base 策略：纯随机选择
"""
import numpy as np
from trainer.base_trainer import BaseTrainer


class RandomStrategy(BaseTrainer):
    """无任何优化策略，作为性能基准"""
    
    def select_clients(self, iter):
        """纯随机选择"""
        m = max(int(self.args.frac * self.args.num_users), 1)
        selected_idxs = np.random.choice(
            range(self.args.num_users), m, replace=False
        ).tolist()
        return selected_idxs