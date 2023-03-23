import time
import pandas
import torch
import torchvision
import modify_googlenet
import numpy as np
from dataloader import *
from utils import *
import pretrainedmodels
from torchvision import datasets, models, transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

debug=False

rate_prob=0.7
target_attack=False

#parameters
batch_size = 25
num_workers = 5 if torch.cuda.is_available() else 0
shuffle = False

num_step=10
step_size=2/255
epsilon=16/255


#load models

#modify the models. to pretrainedmodels.
model_list=[]
# model_list.append(models.inception_v3(pretrained=True,transform_input=True).eval())
model_list.append(pretrainedmodels.resnet152().eval())
model_list.append(pretrainedmodels.resnet101().eval())
model_list.append(pretrainedmodels.resnet50().eval())
# model_list.append(models.densenet121(pretrained=True).eval())
model_list.append(modify_googlenet.modify_vggs(models.vgg16_bn(pretrained=True).eval()))
model_list.append(modify_googlenet.modify_googlenet(models.googlenet(pretrained=True).eval()))
# model_list.append(pretrainedmodels.inceptionv3().eval())
# model_list.append(pretrainedmodels.inceptionv4().eval())
# model_list.append(pretrainedmodels.inceptionresnetv2().eval())
# model_list.append(pretrainedmodels.xception().eval())
for model in model_list:
    model.to(device)
num_models=len(model_list)

#load data
nor=torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
inv_nor = torchvision.transforms.Normalize(
        mean= [-0.485/0.229,-0.456/0.224,-0.406/0.225],
        std= [1/0.229,1/0.224,1/0.225])
transform_test = torchvision.transforms.ToTensor()


dataset=get_imagenet_valset(transform=transform_test)
dataloader=get_imagenet_valloader(dataset,batch_size,shuffle,num_workers)


#I-FGSM  logit-ensemble  CE  untargeted
result=pandas.DataFrame([[0 for i in range(5)] for j in range(6)],['simple','cos','norm','prob','cos+prob','norm+prob'],dtype=float)
result_sum=pandas.DataFrame([[0 for i in range(5)] for j in range(1)],['sum',],dtype=float)
start=time.time()
for batch, (images, labels, targets) in enumerate(dataloader):
    # if batch == 5:
    #     break
    images,labels,targets=images.to(device),labels.to(device),targets.to(device)

    for hold_out in range(num_models):
        hold_out_model=model_list[hold_out]
        ensemble_model_list=model_list.copy()
        ensemble_model_list.remove(hold_out_model)

        delta_sim=torch.zeros_like(images,requires_grad=True).to(device)
        delta_cos=torch.zeros_like(images,requires_grad=True).to(device)
        delta_norm=torch.zeros_like(images,requires_grad=True).to(device)
        delta_prob=torch.zeros_like(images,requires_grad=True).to(device)
        delta_cos_prob=torch.zeros_like(images,requires_grad=True).to(device)
        delta_norm_prob=torch.zeros_like(images,requires_grad=True).to(device)

        # result-sum
        output_ori = hold_out_model(nor(images)).argmax(dim=1)
        if target_attack:
            result_sum[hold_out][0] += (output_ori == labels).sum().item()
        else:
            result_sum[hold_out][0] += (output_ori != targets).sum().item()

        #the feature of original image of all local models
        feature_ori_all = []
        with torch.no_grad():
            for index ,model in enumerate(ensemble_model_list):
                feature_ori_all.append(model.features(nor(images)))
        for step in range(num_step):
            # print(step)

            # simple-ensemble
            adv = nor(images + delta_sim)
            logit_sim = 0.25*ensemble_model_list[0](adv)
            for index, model in enumerate(ensemble_model_list):
                if index == 0:
                    continue
                else:
                    temp = logit_sim + 0.25*model(adv)
                    logit_sim = temp
            prob_sim = sf(logit_sim)
            if target_attack:
                loss = cross_entropy(prob_sim, targets).sum()
            else:
                loss= cross_entropy(prob_sim, labels).sum()
            loss.backward()
            grad = delta_sim.grad.clone().detach()
            delta_sim.grad.zero_()
            if target_attack:
                delta_sim.data = delta_sim.data - step_size * torch.sign(grad)
            else:
                delta_sim.data = delta_sim.data + step_size * torch.sign(grad)

            delta_sim.data = delta_sim.data.clamp(-epsilon, epsilon)
            delta_sim.data = ((images + delta_sim.data).clamp(0, 1)) - images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_sim)).argmax(dim=1)
                if target_attack:
                    result[hold_out][0] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][0] += ((output_ori == labels) == (output_adv != labels)).sum().item()


            # cos
            adv=nor(images+delta_cos)
            feature_adv_all=[]
            logit_all=[]
            prob_all=[]
            weight_all=[]
            for index,model in enumerate(ensemble_model_list):
                logit_now=model(adv)
                logit_all.append(logit_now)
                with torch.no_grad():
                    #calculate the features of adv
                    feature_adv_all.append(model.features(adv))
                    prob_all.append(sf(logit_now))
                    #initialize the tensor for all weight of all images
                    weight_all.append(torch.zeros([len(labels),1]).to(device))
            for num_example in range(len(labels)):#calculate weight for each example
                #get the norm of f_ori-f_adv of all models for the example now
                #for targeted attack, the weight is related to the inverse of prob
                with torch.no_grad():
                    norm_now_all=[]
                    for index,model in enumerate(ensemble_model_list):
                        f1=torch.flatten(feature_adv_all[index][num_example])
                        f2=torch.flatten(feature_ori_all[index][num_example])
                        norm_now_all.append((1.0001+torch.cosine_similarity(f1,f2,dim=0)))
                    #normalize the weight
                    weight_now_sum1=sum(norm_now_all)
                    for index,model in enumerate(ensemble_model_list):
                        weight_all[index][num_example][0]=(norm_now_all[index])/weight_now_sum1
            # calculate ensemble prob
            # print(weight_all)
            logit_dy=logit_all[0]*weight_all[0]
            for index,model in enumerate(ensemble_model_list):
                if index==0:
                    continue
                else:
                    temp=logit_dy+logit_all[index]*weight_all[index]
                    logit_dy=temp
            prob_dy=sf(logit_dy)
            if target_attack:
                loss=cross_entropy(prob_dy,targets).sum()
            else:
                loss=cross_entropy(prob_dy,labels).sum()
            loss.backward()
            grad=delta_cos.grad.clone().detach()
            delta_cos.grad.zero_()
            if target_attack:
                delta_cos.data=delta_cos.data - step_size*torch.sign(grad)
            else:
                delta_cos.data=delta_cos.data + step_size*torch.sign(grad)
            delta_cos.data=delta_cos.data.clamp(-epsilon,epsilon)
            delta_cos.data=((images+delta_cos.data).clamp(0,1))-images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_cos)).argmax(dim=1)
                if target_attack:
                    result[hold_out][1] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][1] += ((output_ori == labels) == (output_adv != labels)).sum().item()

            # norm
            adv=nor(images+delta_norm)
            feature_adv_all=[]
            logit_all=[]
            prob_all=[]
            weight_all=[]
            for index,model in enumerate(ensemble_model_list):
                logit_now=model(adv)
                logit_all.append(logit_now)
                with torch.no_grad():
                    #calculate the features of adv
                    feature_adv_all.append(model.features(adv))
                    prob_all.append(sf(logit_now))
                    #initialize the tensor for all weight of all images
                    weight_all.append(torch.zeros([len(labels),1]).to(device))
            for num_example in range(len(labels)):#calculate weight for each example
                #get the norm of f_ori-f_adv of all models for the example now
                #for targeted attack, the weight is related to the inverse of prob
                with torch.no_grad():
                    norm_now_all=[]
                    for index,model in enumerate(ensemble_model_list):
                        f1=torch.norm(feature_adv_all[index][num_example],p=2)
                        f2=torch.norm(feature_ori_all[index][num_example],p=2)
                        norm_now_all.append(0.0001+torch.abs((f1-f2)))
                    #normalize the weight
                    if debug:
                        print(norm_now_all)
                        print('8888888888888888')
                    weight_now_sum1=sum(norm_now_all)
                    for index,model in enumerate(ensemble_model_list):
                        weight_all[index][num_example][0]=(norm_now_all[index])/weight_now_sum1
            # calculate ensemble prob
            # print(weight_all)
            if debug:
                print(weight_all)
                print('8888888888888888')
            logit_dy=logit_all[0]*weight_all[0]
            for index,model in enumerate(ensemble_model_list):
                if index==0:
                    continue
                else:
                    temp=logit_dy+logit_all[index]*weight_all[index]
                    logit_dy=temp
            prob_dy=sf(logit_dy)
            if target_attack:
                loss=cross_entropy(prob_dy,targets).sum()
            else:
                loss=cross_entropy(prob_dy,labels).sum()
            loss.backward()
            grad=delta_norm.grad.clone().detach()
            delta_norm.grad.zero_()
            if target_attack:
                delta_norm.data=delta_norm.data - step_size*torch.sign(grad)
            else:
                delta_norm.data=delta_norm.data + step_size*torch.sign(grad)
            delta_norm.data=delta_norm.data.clamp(-epsilon,epsilon)
            delta_norm.data=((images+delta_norm.data).clamp(0,1))-images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_norm)).argmax(dim=1)
                if target_attack:
                    result[hold_out][2] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][2] += ((output_ori == labels) == (output_adv != labels)).sum().item()



            # prob
            adv=nor(images+delta_prob)
            feature_adv_all=[]
            logit_all=[]
            prob_all=[]
            weight_all=[]
            for index,model in enumerate(ensemble_model_list):
                logit_now=model(adv)
                logit_all.append(logit_now)
                with torch.no_grad():
                    #calculate the features of adv
                    feature_adv_all.append(model.features(adv))
                    prob_all.append(sf(logit_now))
                    #initialize the tensor for all weight of all images
                    weight_all.append(torch.zeros([len(labels),1]).to(device))
            for num_example in range(len(labels)):#calculate weight for each example
                #get the norm of f_ori-f_adv of all models for the example now
                #for targeted attack, the weight is related to the inverse of prob
                with torch.no_grad():
                    prob_now_all=[]
                    for index,model in enumerate(ensemble_model_list):
                        if target_attack:
                            prob_now_all.append(1/(prob_all[index][num_example][targets[num_example].item()].item()))
                        else:
                            prob_now_all.append(prob_all[index][num_example][labels[num_example].item()].item())
                    #normalize the weight
                    weight_now_sum2=sum(prob_now_all)
                    for index,model in enumerate(ensemble_model_list):
                        weight_all[index][num_example][0]=(prob_now_all[index])/weight_now_sum2
            # calculate ensemble prob
            # print(weight_all)
            logit_dy = logit_all[0] * weight_all[0]
            for index, model in enumerate(ensemble_model_list):
                if index == 0:
                    continue
                else:
                    temp = logit_dy + logit_all[index] * weight_all[index]
                    logit_dy = temp
            prob_dy = sf(logit_dy)
            if target_attack:
                loss=cross_entropy(prob_dy,targets).sum()
            else:
                loss=cross_entropy(prob_dy,labels).sum()
            loss.backward()
            grad=delta_prob.grad.clone().detach()
            delta_prob.grad.zero_()
            if target_attack:
                delta_prob.data=delta_prob.data - step_size*torch.sign(grad)
            else:
                delta_prob.data=delta_prob.data + step_size*torch.sign(grad)
            delta_prob.data=delta_prob.data.clamp(-epsilon,epsilon)
            delta_prob.data=((images+delta_prob.data).clamp(0,1))-images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_prob)).argmax(dim=1)
                if target_attack:
                    result[hold_out][3] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][3] += ((output_ori == labels) == (output_adv != labels)).sum().item()





            # cos+prob
            adv=nor(images+delta_cos_prob)
            feature_adv_all=[]
            logit_all=[]
            prob_all=[]
            weight_all=[]
            for index,model in enumerate(ensemble_model_list):
                logit_now=model(adv)
                logit_all.append(logit_now)
                with torch.no_grad():
                    #calculate the features of adv
                    feature_adv_all.append(model.features(adv))
                    prob_all.append(sf(logit_now))
                    #initialize the tensor for all weight of all images
                    weight_all.append(torch.zeros([len(labels),1]).to(device))
            for num_example in range(len(labels)):#calculate weight for each example
                #get the norm of f_ori-f_adv of all models for the example now
                #for targeted attack, the weight is related to the inverse of prob
                with torch.no_grad():
                    prob_now_all=[]
                    norm_now_all=[]
                    for index,model in enumerate(ensemble_model_list):
                        f1=torch.flatten(feature_adv_all[index][num_example])
                        f2=torch.flatten(feature_ori_all[index][num_example])
                        norm_now_all.append((1.0001+torch.cosine_similarity(f1,f2,dim=0)))
                        if target_attack:
                            prob_now_all.append(1/(prob_all[index][num_example][targets[num_example].item()].item()))
                        else:
                            prob_now_all.append(prob_all[index][num_example][labels[num_example].item()].item())
                    #normalize the weight
                    weight_now_sum1=sum(norm_now_all)
                    weight_now_sum2=sum(prob_now_all)
                    for index,model in enumerate(ensemble_model_list):
                        weight_all[index][num_example][0]=rate_prob*((prob_now_all[index])/weight_now_sum2)+(1-rate_prob)*((norm_now_all[index])/weight_now_sum1)
            # calculate ensemble prob
            # print(weight_all)
            logit_dy = logit_all[0] * weight_all[0]
            for index, model in enumerate(ensemble_model_list):
                if index == 0:
                    continue
                else:
                    temp = logit_dy + logit_all[index] * weight_all[index]
                    logit_dy = temp
            prob_dy = sf(logit_dy)
            if target_attack:
                loss=cross_entropy(prob_dy,targets).sum()
            else:
                loss=cross_entropy(prob_dy,labels).sum()
            loss.backward()
            grad=delta_cos_prob.grad.clone().detach()
            delta_cos_prob.grad.zero_()
            if target_attack:
                delta_cos_prob.data=delta_cos_prob.data - step_size*torch.sign(grad)
            else:
                delta_cos_prob.data=delta_cos_prob.data + step_size*torch.sign(grad)
            delta_cos_prob.data=delta_cos_prob.data.clamp(-epsilon,epsilon)
            delta_cos_prob.data=((images+delta_cos_prob.data).clamp(0,1))-images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_cos_prob)).argmax(dim=1)
                if target_attack:
                    result[hold_out][4] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][4] += ((output_ori == labels) == (output_adv != labels)).sum().item()

            # norm+prob
            adv=nor(images+delta_norm_prob)
            feature_adv_all=[]
            logit_all=[]
            prob_all=[]
            weight_all=[]
            for index,model in enumerate(ensemble_model_list):
                logit_now=model(adv)
                logit_all.append(logit_now)
                with torch.no_grad():
                    #calculate the features of adv
                    feature_adv_all.append(model.features(adv))
                    prob_all.append(sf(logit_now))
                    #initialize the tensor for all weight of all images
                    weight_all.append(torch.zeros([len(labels),1]).to(device))
            for num_example in range(len(labels)):#calculate weight for each example
                #get the norm of f_ori-f_adv of all models for the example now
                #for targeted attack, the weight is related to the inverse of prob
                with torch.no_grad():
                    prob_now_all=[]
                    norm_now_all=[]
                    for index,model in enumerate(ensemble_model_list):
                        f1=torch.norm(feature_adv_all[index][num_example],p=2)
                        f2=torch.norm(feature_ori_all[index][num_example],p=2)
                        norm_now_all.append(0.0001+torch.abs((f1-f2)))
                        if target_attack:
                            prob_now_all.append(1/(prob_all[index][num_example][targets[num_example].item()].item()))
                        else:
                            prob_now_all.append(prob_all[index][num_example][labels[num_example].item()].item())
                    #normalize the weight
                    weight_now_sum1=sum(norm_now_all)
                    weight_now_sum2=sum(prob_now_all)
                    if debug:
                        print(norm_now_all)
                    for index,model in enumerate(ensemble_model_list):
                        weight_all[index][num_example][0]=rate_prob*((prob_now_all[index])/weight_now_sum2)+(1-rate_prob)*((norm_now_all[index])/weight_now_sum1)
            # calculate ensemble prob
            # print(weight_all)
            if debug:
                print(weight_all)
            logit_dy = logit_all[0] * weight_all[0]
            for index, model in enumerate(ensemble_model_list):
                if index == 0:
                    continue
                else:
                    temp = logit_dy + logit_all[index] * weight_all[index]
                    logit_dy = temp
            prob_dy = sf(logit_dy)
            if target_attack:
                loss=cross_entropy(prob_dy,targets).sum()
            else:
                loss=cross_entropy(prob_dy,labels).sum()
            loss.backward()
            grad=delta_norm_prob.grad.clone().detach()
            delta_norm_prob.grad.zero_()
            if target_attack:
                delta_norm_prob.data=delta_norm_prob.data - step_size*torch.sign(grad)
            else:
                delta_norm_prob.data=delta_norm_prob.data + step_size*torch.sign(grad)
            delta_norm_prob.data=delta_norm_prob.data.clamp(-epsilon,epsilon)
            delta_norm_prob.data=((images+delta_norm_prob.data).clamp(0,1))-images
            if step == num_step-1:
                output_adv=hold_out_model(nor(images+delta_norm_prob)).argmax(dim=1)
                if target_attack:
                    result[hold_out][5] += ((output_ori != targets) == (output_adv == targets)).sum().item()
                else:
                    result[hold_out][5] += ((output_ori == labels) == (output_adv != labels)).sum().item()



    print('[{} / {}]time:{}min'.format(batch, 1000/batch_size, (time.time() - start) / 60))
    if debug:
        break

print(result)
print(result_sum)
for hold_out in range(num_models):
    for i in range(6):
        result[hold_out][i]=result[hold_out][i]/result_sum[hold_out][0]

print(result)
if target_attack:
    result.to_csv('1_logit_target.csv')
else:
    result.to_csv('1_logit_untarget.csv')