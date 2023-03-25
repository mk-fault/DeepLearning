import sys

import torch
from torch import nn,optim
from torchvision import datasets,transforms
import json
import os
import matplotlib.pyplot as plt
from torchvision.models import ResNet34_Weights,ResNet101_Weights
from model import resnet34,resnet101
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("training on {}".format(device))

    # 数据预处理
    data_transform = {
        "train" : transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])]),
        "val" : transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])}


    # 数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    # image_path = data_root + "/data/flower_data/"
    image_path = os.path.join(data_root,'data','flower_data')
    assert os.path.exists(image_path),'{},not exist'.format(image_path)

    # 训练集生成
    train_dataset = datasets.ImageFolder(os.path.join(image_path,"train"),transform=data_transform['train'])
    train_num = len(train_dataset)

    # 将类别映射到序号，并写入JSON中
    #{'daisy':0,'dandelion':1,'rose':2,'sunflower':3,'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in flower_list.items())
    #write dict into json file
    json_str = json.dumps(cla_dict,indent=4)
    with open('class_indices.json','w') as f:
        f.write(json_str)

    batch_size = 64
    num_worker = min(os.cpu_count(),batch_size)
    print("num_worker:{}".format(num_worker))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_worker)

    # 验证集
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path,'val'),transform=data_transform['val'])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_worker)

    print("training pic:{},validate pic:{}".format(train_num,val_num))

    # 载入模型和权重
    # net = resnet101()
    # net.load_state_dict(torch.load('./resnet101-63fe2227.pth'),strict=False)

    # 修改全连接层
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel,5)

    net = resnet101(5)
    writer = SummaryWriter(log_dir='./logs')
    init_tensor = torch.zeros((1,3,224,224))
    writer.add_graph(net,init_tensor)
    net.to(device)

    # 定义参数
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params,lr=0.0001)
    epochs = 50
    best_acc = 0
    shape_f = 0
    save_path = '../../source/resnet101.pth'
    train_steps = len(train_loader)

    # 开始训练
    tag = ['training loss', 'validate accuracy']
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,file=sys.stdout)
        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits,labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:3f}".format(epoch + 1,epochs,loss)

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader,file=sys.stdout)
            for val_data in val_bar:
                val_images,val_labels = val_data
                outputs = net(val_images.to(device))
                if shape_f == 0:
                    print(outputs.shape)
                    shape_f += 1
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' % (epoch + 1,running_loss / train_steps,val_accurate))
        writer.add_scalar(tag[0],running_loss / train_steps,epoch)
        writer.add_scalar(tag[1],val_accurate,epoch)



        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),save_path)

    print('Finished')
    writer.close()

if __name__ == '__main__':
    main()