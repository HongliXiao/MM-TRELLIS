from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval,  **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs) # torch.Size([1, 8, 16, 16, 16])
            neg_pred = super()._inference_model(model, x_t, t, neg_cond,  **kwargs) # torch.Size([1, 8, 16, 16, 16])
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)


    # def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
    #     if cfg_interval[0] <= t <= cfg_interval[1]:
    #         pred = super()._inference_model(model, x_t, t, cond, **kwargs) # torch.Size([1, 8, 16, 16, 16])
    #         neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs) # torch.Size([1, 8, 16, 16, 16])
    #         return (1 + cfg_strength) * pred - cfg_strength * neg_pred, pred,  cfg_strength * neg_pred
    #     else:
    #         pred = super()._inference_model(model, x_t, t, cond, **kwargs)
    #         return pred, pred, pred


    # def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
      
    #     return super()._inference_model(model, x_t, t, neg_cond, **kwargs)


    # def _inference_model_wo_cfg(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
    #     # if cfg_interval[0] <= t <= cfg_interval[1]:
    #     #     pred = super()._inference_model(model, x_t, t, cond, **kwargs) # torch.Size([1, 8, 16, 16, 16])
    #     #     neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs) # torch.Size([1, 8, 16, 16, 16])
    #     #     return (1 + cfg_strength) * pred - cfg_strength * neg_pred
    #     # else:
    #     #     return super()._inference_model(model, x_t, t, cond, **kwargs)
    #     return super()._inference_model(model, x_t, t, cond, **kwargs)

