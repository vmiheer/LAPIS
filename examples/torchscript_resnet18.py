# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

from PIL import Image
import requests

import torch
import torchvision.models as models
from torchvision import transforms

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from kokkos_mlir.linalg_kokkos_backend import KokkosBackend

from timeit import default_timer as timer

def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(idx, labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3

def predictions(torch_func, jit_func, kokkos_func, img, labels):
    t1 = timer()
    raw_pred = torch_func(img)
    t2 = timer()
    print("Torch time: ", t2 - t1)
    print("Gold Top-3 probs: ", raw_pred[0][208], ", ", raw_pred[0][207], ", ", raw_pred[0][176])
    #golden_prediction = top3_possibilities(raw_pred)
    #print("PyTorch prediction")
    #print(golden_prediction)
    t1 = timer()
    raw_pred = torch.from_numpy(jit_func(img.numpy()))
    t2 = timer()
    print("MLIR JIT time: ", t2 - t1)
    #prediction = top3_possibilities(raw_pred)
    #print("torch-mlir prediction")
    #print(prediction)
    t1 = timer()
    raw_pred = torch.from_numpy(kokkos_func(img.numpy()))
    t2 = timer()
    print("Kokkos time: ", t2 - t1)
    #print("predictions shape: ", pred.shape)
    print("Kokkos Top-3 probs: ", raw_pred[0][208], ", ", raw_pred[0][207], ", ", raw_pred[0][176])
    #torchpred = torch.from_numpy(pred)
    #_, indexes = torch.sort(torchpred, descending=True)
    #top3 = [(idx, labels[idx], torchpred[0][idx]) for idx in indexes[0][:3]]
    #print(top3)

image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True)
resnet18.train(False)
module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="linalg-on-tensors")
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)

#
#predictions(resnet18.forward, jit_module.forward, img, labels)

# Compile module to mid-level MLIR (lower from linalg/tensor to memrefs, parallel for, arithmetic)

kModule = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
kBackend = KokkosBackend.KokkosBackendLinalgOnTensorsBackend()
kCompiledModule = kBackend.compile(kModule)
predictions(resnet18.forward, jit_module.forward, kCompiledModule.forward, img, labels)

#print("Dump of Kokkos module:")
#kCompiled.dump()
#print("Args of body of Kokkos module:")
#print(dir(kCompiled.body.arguments))
#print("body: ")
#print(dir(kCompiled.body))
#print("region.blocks: ")
#print(dir(kCompiled.body.region.blocks))
#print("operations: ")
#print(dir(kCompiled.body.operations))

#def recursiveDisplayOperation(op, indent):
#    print(" " * 4 * indent, op)
#    if 'operations' in dir(op):
#        for subOp in op.operations:
#            recursiveDisplayOperation(subOp, indent + 1)

#print("Module body:")
#recursiveDisplayOperation(kCompiled.body, 0)

