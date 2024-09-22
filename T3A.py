# Section 3.D Test-Time Distribution Alignment
# Adapted from T3A (https://github.com/matsuolab/T3A)
# Key difference from T3A: We introduce an interpolation coefficient α and keep the original bias from the classifier to avoid overly aggressive updates, which has been shown to be beneficial in the model extraction scenario.

import torch

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    
class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        self.warmup_supports = self.classifier.weight.data.clone().detach()
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.clone().detach()
        self.labels = self.warmup_labels.clone().detach()
        self.ent = self.warmup_ent.clone().detach()

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

        self.alpha = hparams.get('alpha', 0.5)

    def forward(self, x, adapt=False, interpolation=False):
        z = self.featurizer(x)
        z = z.view(z.size(0), -1)
        
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ labels)
        
        if interpolation:
            final_weights = self.alpha * torch.nn.functional.normalize(weights, dim=0) + (1 - self.alpha) * self.classifier.weight.T
        else:
            final_weights = self.classifier.weight.T
        
        return z @ final_weights + self.classifier.bias

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K

        indices1 = torch.arange(len(ent_s), device=self.ent.device)

        indices = []
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False, interpolation=False):
        return self(x, adapt, interpolation)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data