#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('wget -N https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/cifar10.zip')
get_ipython().system('unzip -o cifar10.zip -d ./datasets')
get_ipython().system('tree ./datasets/cifar10')


# In[3]:


import mindspore.nn as nn
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore import context
import numpy as np
import matplotlib.pyplot as plt

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def create_dataset(data_home, repeat_num=1, batch_size=32, do_train=True, device_target="CPU"):
    """
    create data for next use such as training or inferring
    """

    cifar_ds = ds.Cifar10Dataset(data_home,num_parallel_workers=8, shuffle=True)

    c_trans = []
    if do_train:
        c_trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    c_trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=8)

    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds


ds_train_path = "./datasets/cifar10/train/"
dataset_show = create_dataset(ds_train_path)
with open(ds_train_path+"batches.meta.txt","r",encoding="utf-8") as f:
    all_name = [name.replace("\n","") for name in f.readlines()]

iterator_show= dataset_show.create_dict_iterator()
dict_data = iterator_show.get_next()
images = dict_data["image"].asnumpy()
labels = dict_data["label"].asnumpy()
count = 1
get_ipython().run_line_magic('matplotlib', 'inline')
for i in images:
    plt.subplot(4, 8, count)
    # Images[0].shape is (3,224,224).We need transpose as (224,224,3) for using in plt.show().
    picture_show = np.transpose(i,(1,2,0))
    picture_show = picture_show/np.amax(picture_show)
    picture_show = np.clip(picture_show, 0, 1)
    plt.title(all_name[labels[count-1]])
    picture_show = np.array(picture_show,np.float32)
    plt.imshow(picture_show)
    count += 1
    plt.axis("off")

print("The dataset size is:", dataset_show.get_dataset_size())
print("The batch tensor is:",images.shape)
plt.show()


# In[4]:


get_ipython().system('wget -N https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/source-codes/resnet.py')


# In[5]:


from resnet import resnet50

net = resnet50(batch_size=32, num_classes=10)


# In[6]:


import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits

ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)


# In[7]:


from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import os
from mindspore import Model


model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})
# As for train, users could use model.train

epoch_size = 10
ds_train_path = "./datasets/cifar10/train/"
model_path = "./models/ckpt/mindspore_vision_application/"
os.system('rm -f {0}*.ckpt {0}*.meta {0}*.pb'.format(model_path))

dataset = create_dataset(ds_train_path )
batch_num = dataset.get_dataset_size()
config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=35)
ckpoint_cb = ModelCheckpoint(prefix="train_resnet_cifar10", directory=model_path, config=config_ck)
loss_cb = LossMonitor(142)
model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])


# In[8]:


get_ipython().system('tree ./models/ckpt/mindspore_vision_application/')


# In[9]:


from resnet import resnet50

from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = resnet50(batch_size=32, num_classes=10)
# return a parameter dict for model
param_dict = load_checkpoint("./models/ckpt/mindspore_vision_application/train_resnet_cifar10-10_1562.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50_Jack20.mindir', file_format='MINDIR')


# In[8]:


# As for evaluation, users could use model.eval
ds_eval_path = "./datasets/cifar10/test/"
eval_dataset = create_dataset(ds_eval_path, do_train=False)
res = model.eval(eval_dataset)
print("result: ", res)

