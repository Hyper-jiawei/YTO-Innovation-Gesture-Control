import bootstrap
import json
import platform
import sys
from pathlib import Path
import importlib.metadata as metadata
import importlib.util


def check():
    report=dict(python=sys.executable,version=sys.version,platform=platform.platform(),packages={})
    for package in ['mediapipe','opencv-contrib-python','numpy','torch','torchvision','ultralytics','zxing-cpp']:
        try: report['packages'][package]=metadata.version(package)
        except metadata.PackageNotFoundError: report['packages'][package]=None
    try:
        import torch
        report.update(torch_path=torch.__file__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),architectures=torch.cuda.get_arch_list())
            x=torch.randn((512,512),device='cuda')
            report['cuda_kernel_result']=(x@x).mean().item()
            # Validate backward kernels as well, for the supplied training workflow.
            layer=torch.nn.Linear(63,11).cuda()
            layer(torch.randn(32,63,device='cuda')).sum().backward()
            report['cuda_backward']=True
        else: report['cuda_backward']=False
    except Exception as e: report['torch_error']=repr(e)
    report['mediapipe_backend']='CPU: Windows Python Tasks; independent of PyTorch CUDA'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    out=bootstrap.ROOT/'outputs/environment.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__': check()
