__all__ = ['Dataset']


from .. import utils


class Dataset(object):
    def __init__(self, data, target, metric):
        """dataset for metric calculation

        Args:
            data (np.ndarray, torch.Tensor, tuple or list): input data for pytorch model and trt engine inference, will
                be automatically transformed to numpy data.
            target (np.ndarray, torch.Tensor, tuple or list): target for metric calculation, it will be automatically
                transformed to numpy data
            metric (vedadep.benchmark.metric.BaseMetric): metric for calculation between pred and target.
        """

        self.data = utils.to(data, 'numpy')
        self.target = utils.to(target, 'numpy')
        self.metric_obj = metric

    def metric(self, pred):
        return self.metric_obj.metric(pred, self.target)

    @property
    def metric_name(self):
        return self.metric_obj.metric_name()
