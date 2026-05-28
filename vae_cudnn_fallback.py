import logging

import torch

import comfy.model_management


log = logging.getLogger(__name__)

_CUDNN_ENGINE_ERROR = "GET was unable to find an engine to execute this computation"


def is_cudnn_engine_error(exc):
    return _CUDNN_ENGINE_ERROR in str(exc)


def _get_comfy_ops():
    try:
        import comfy.ops as comfy_ops
    except Exception:
        return None
    return comfy_ops


def _has_conv_workaround(comfy_ops):
    return comfy_ops is not None and hasattr(comfy_ops, "NVIDIA_MEMORY_CONV_BUG_WORKAROUND")


def _is_lightricks_video_vae(vae):
    first_stage_model = getattr(vae, "first_stage_model", None)
    module_name = first_stage_model.__class__.__module__ if first_stage_model is not None else ""
    return module_name.startswith("comfy.ldm.lightricks.")


def _run_with_conv_workaround_disabled(operation, comfy_ops):
    old_workaround = comfy_ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND
    try:
        comfy_ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
        return operation()
    finally:
        comfy_ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = old_workaround


def _retry_with_cudnn_workarounds(operation, label, prefer_disabled_conv_workaround=False):
    comfy_ops = _get_comfy_ops()
    tried_disabled_conv_workaround = False
    last_exc = None

    if (
        prefer_disabled_conv_workaround
        and _has_conv_workaround(comfy_ops)
        and comfy_ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND
    ):
        try:
            tried_disabled_conv_workaround = True
            return _run_with_conv_workaround_disabled(operation, comfy_ops)
        except RuntimeError as exc:
            if not is_cudnn_engine_error(exc):
                raise
            last_exc = exc
            comfy.model_management.soft_empty_cache()
    else:
        try:
            return operation()
        except RuntimeError as exc:
            if not is_cudnn_engine_error(exc):
                raise
            last_exc = exc
            comfy.model_management.soft_empty_cache()

    if (
        not tried_disabled_conv_workaround
        and _has_conv_workaround(comfy_ops)
        and comfy_ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND
    ):
        try:
            log.warning(
                "%s hit a cuDNN Conv3D engine error; retrying with "
                "ComfyUI's NVIDIA Conv3D workaround disabled.",
                label,
            )
            return _run_with_conv_workaround_disabled(operation, comfy_ops)
        except RuntimeError as retry_exc:
            if not is_cudnn_engine_error(retry_exc):
                raise
            last_exc = retry_exc

    if torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled:
        old_cudnn_enabled = torch.backends.cudnn.enabled
        try:
            log.warning(
                "%s still hit a cuDNN Conv3D engine error; retrying once "
                "with cuDNN disabled.",
                label,
            )
            torch.backends.cudnn.enabled = False
            comfy.model_management.soft_empty_cache()
            return operation()
        finally:
            torch.backends.cudnn.enabled = old_cudnn_enabled

    raise last_exc


def patch_vae_cudnn_fallback():
    import comfy.sd

    vae_cls = comfy.sd.VAE
    if getattr(vae_cls, "_wdc_cudnn_fallback_patched", False):
        return

    original_encode = vae_cls.encode
    original_decode = vae_cls.decode

    def encode_with_fallback(self, pixel_samples):
        return _retry_with_cudnn_workarounds(
            lambda: original_encode(self, pixel_samples),
            "VAE encode",
            prefer_disabled_conv_workaround=_is_lightricks_video_vae(self),
        )

    def decode_with_fallback(self, samples_in, vae_options={}):
        return _retry_with_cudnn_workarounds(
            lambda: original_decode(self, samples_in, vae_options),
            "VAE decode",
            prefer_disabled_conv_workaround=_is_lightricks_video_vae(self),
        )

    vae_cls.encode = encode_with_fallback
    vae_cls.decode = decode_with_fallback
    vae_cls._wdc_cudnn_fallback_patched = True
    vae_cls._wdc_original_encode = original_encode
    vae_cls._wdc_original_decode = original_decode
