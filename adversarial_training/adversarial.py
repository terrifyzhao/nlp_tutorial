import torch


class AT:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, emb_name='emb.'):
        """
        备份embedding matrix 并添加我们的扰动项
        :param emb_name: embedding层的名字
        """
        raise NotImplemented

    def restore(self, emb_name='emb.'):
        """
        把embedding matrix的参数恢复
        :param emb_name: embedding层的名字
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM(AT):
    def attack(self, epsilon=1., emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


class FGSM(AT):
    def attack(self, epsilon=1., emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * torch.sign(param.grad)
                    param.data.add_(r_at)


class FreeAT(AT):

    def __init__(self, model):
        super().__init__(model)
        self.grad_backup = {}

    def attack(self, epsilon=0.3, alpha=0.01, emb_name='emb.', first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if first_attack:
                    self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 得到新的扰动
                    r_at = alpha * param.grad / norm
                    r_at = torch.clamp(r_at, - epsilon, epsilon)
                    # 加到输入上
                    param.data.add_(r_at)

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class FreeLB(AT):

    def __init__(self, model):
        super().__init__(model)
        self.grad_backup = {}

    def attack(self, epsilon=0.01, alpha=5e-3, emb_name='emb.', first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if first_attack:
                    r_at = torch.Tensor(1).uniform_(-epsilon, epsilon)
                else:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = alpha * param.grad / norm
                        r_at = torch.clamp(r_at, - epsilon, epsilon)
                param.data.add_(r_at)

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
