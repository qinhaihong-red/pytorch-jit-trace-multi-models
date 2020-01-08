import numpy as np
import torch
from typing import List,Tuple

def gen_traced_model(model,device,inputs_shape,saved_name,check):
    if not isinstance(inputs_shape,list):
        raise TypeError('inputs_shape must be list type.')
    model.eval()
    model.to(device)
    inputs_num=len(inputs_shape)
    inputs=[]
    inputs_check=[]
    for i in range(inputs_num):
        inputs.append(torch.randn(inputs_shape[i]))
        inputs_check.append(torch.randn(inputs_shape[i]))


    with torch.no_grad():
        #out=model(inputs)
        if not check:
            traced_model=torch.jit.trace(model,tuple(inputs),check_trace=False)
        else:
            traced_model=torch.jit.trace(model,tuple(inputs),check_trace=True,check_inputs=[tuple(inputs_check)])

        #out=traced_model(inputs)
        #print(out.shape)
        
        traced_model.save(saved_name+'.pt')
        print('traced ok.')
        return traced_model


def gen_cpu_traced_model(model,inputs_shape,output_name,check=True):
    if check:
        saved_name=output_name+'_cpu_checked'
    else:
        saved_name=output_name+'_cpu_nochecked'
    return gen_traced_model(model,torch.device('cpu'),inputs_shape,saved_name,check)


def gen_gpu_traced_model(model,inputs_shape:List,output_name,check=True):
    if not torch.cuda.is_available():
        raise RuntimeError('cuda not supported.')

    if check:
        saved_name=output_name+'_gpu_checked'
    else:
        saved_name=output_name+'_gpu_nochecked'
    return gen_traced_model(model,torch.device('cuda'),inputs_shape,saved_name,check)