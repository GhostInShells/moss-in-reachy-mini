import os
import os.path as osp

import onnxruntime
from insightface.model_zoo.model_zoo import find_onnx_file, ModelRouter, get_default_providers, \
    get_default_provider_options
from insightface.utils import download_onnx


def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    allow_download = kwargs.get('download', False)
    download_zip = kwargs.get('download_zip', False)
    if not name.endswith('.onnx'):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    if not osp.exists(model_file) and allow_download:
        model_file = download_onnx('models', model_file, root=root, download_zip=download_zip)
    assert osp.exists(model_file), 'model_file %s should exist'%model_file
    assert osp.isfile(model_file), 'model_file %s should be a file'%model_file
    router = ModelRouter(model_file)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    sess_options = onnxruntime.SessionOptions()
    # 关闭CPU内存池，以减少内存占用
    sess_options.enable_cpu_mem_arena = False
    # 可选：关闭内存模式优化，可能进一步降低内存使用
    sess_options.enable_mem_pattern = False
    # sess_options.enable_mem_reuse = False
    model = router.get_model(providers=providers, provider_options=provider_options, sess_options=sess_options)
    return model