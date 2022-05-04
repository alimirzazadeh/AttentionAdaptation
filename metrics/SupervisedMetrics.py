# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:10 2021

@author: alimi
"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from ipdb import set_trace as bp

class Evaluator:
    def __init__(self):
        self.supervised_losses_source = []
        self.supervised_losses_target = []
        # self.accuracies = []
        self.unsupervised_losses_source = []
        self.unsupervised_losses_target = []
        self.accuracy_source = []
        self.accuracy_target = []
        self.best_accuracy_source = 0
        self.best_accuracy_target = 0
        self.bestUnsupSum = 999999
        self.counter = 0




    def evaluateModelSupervisedPerformance(self, model, testloader, criteron, device, optimizer, storeLoss = False, batchDirectory='', source_or_target=0):
        #model.eval()
        running_corrects = 0
        running_loss = 0.0
        
        tp = None
        fp = None
        fn = None
        firstTime = True
        allTrueLabels = None
        allPredLabels = None
        
        datasetSize = len(testloader.dataset)
        


        with torch.set_grad_enabled(False):
            m = nn.Sigmoid()
            for i, data in enumerate(testloader, 0):
                #bp()
                # print(i)
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) 
                #print(outputs)
                #print(labels)
                #print(outputs.shape)
                #print(labels.shape)
                #bp()
                
                l1 = criteron(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += l1.item()
                
                if firstTime:
                    allTrueLabels = labels.cpu().detach().numpy()
                    allPredLabels = m(outputs).cpu().detach().numpy()
                    firstTime = False
                else:
                    allTrueLabels = np.append(allTrueLabels, labels.cpu().detach().numpy(), axis=0)
                    allPredLabels = np.append(allPredLabels, outputs.cpu().detach().numpy(), axis=0)

                    
                #bp()
                running_corrects += torch.sum(preds == labels.data)

                
                # for pred in range(preds.shape[0]):
                #     running_corrects += labels[pred, int(preds[pred])]
                #     m = nn.Sigmoid()
                #     pred_probability = m(outputs[pred])
                #     pred_logits = (pred_probability > 0.5).int()
                    
                #     if tp == None:
                #         tp = (pred_logits + labels[pred] > 1).int()
                #         fp = (torch.subtract(pred_logits, labels[pred]) > 0).int()
                #         fn = (torch.subtract(pred_logits, labels[pred]) < 0).int()
                #     else:
                #         tp += (pred_logits + labels[pred] > 1).int()
                #         fp += (torch.subtract(pred_logits, labels[pred]) > 0).int()
                #         fn += (torch.subtract(pred_logits, labels[pred]) < 0).int()
                    
                    # if labels[pred, int(preds[pred])] == 1:
                    #     tp += 1
                    # else:
                    #     fp += 1
                    # fn += 
                    # print(labels[pred, int(preds[pred])])
                # print(running_corrects.item())
                # del l1, inputs, labels, outputs, preds
            # print('\n Test Model Accuracy: %.3f' % float(running_corrects.item() / datasetSize))
            supervised_loss = float(running_loss / datasetSize)
            print('\n Test Model Supervised Loss: %.3f' % supervised_loss)
            #bp()
            #mAP = average_precision_score(allTrueLabels,allPredLabels,average='weighted')
            accuracy = float(running_corrects.item() / datasetSize)
            if source_or_target == 0:
                print('\n Source Data Model accuracy: %.3f' % accuracy)
            else:
                print('\n Target Data Model accuracy: %.3f' % accuracy)
            # f1_score = self.calculateF1score(tp, fp, fn)
            
            # try:
            #     pd.DataFrame(dict(enumerate(f1_score.data.cpu().numpy())),index=[self.counter]).to_csv(batchDirectory+'saved_figs/f1_scores.csv', mode='a', header=False)
            # except:
            #     pd.DataFrame(dict(enumerate(f1_score.data.cpu().numpy())),index=[self.counter]).to_csv(batchDirectory+'saved_figs/f1_scores.csv', header=False)
            self.counter += 1
            
            # f1_sum = np.nansum(f1_score.data.cpu().numpy())
            
            if source_or_target == 0 and accuracy > self.best_accuracy_source:
                self.best_accuracy_source = accuracy
                print("\n Best Source Accuracy so far: ", self.best_accuracy_source)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, source_sup_target=0)
                # self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory)
            elif source_or_target == 1 and accuracy > self.best_accuracy_target:
                self.best_accuracy_target = accuracy
                print("\n Best Target Accuracy so far: ", self.best_accuracy_target)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, source_sup_target=2)
                # self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory)

            
            # print('\n F1 Score: ', f1_score.data.cpu().numpy())
            # print('\n F1 Score Sum: ', f1_sum)
                    
        
            
            if storeLoss:
                
                # self.accuracies.append(float(running_corrects.item() / datasetSize))
                if source_or_target == 0:
                    self.supervised_losses_source.append(supervised_loss)
                    self.accuracy_source.append(accuracy)
                else:
                    self.supervised_losses_target.append(supervised_loss)
                    self.accuracy_target.append(accuracy)
        #print('..')
    def evaluateModelUnsupervisedPerformance(self, model, testloader, CAMLossInstance, device, optimizer, target_category=None, storeLoss = False, batchDirectory='', source_or_target=0):
        # model.eval()
        running_loss = 0.0
        datasetSize = len(testloader.dataset)
        #print('.')
        with torch.set_grad_enabled(True):
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                l1 = CAMLossInstance(inputs, target_category, visualize=False)
                running_loss += l1.item()
        unsupervised_loss = float(running_loss / datasetSize)
        print('\n Test Model Unsupervised Loss: %.3f' % unsupervised_loss)

        if source_or_target == 1 and unsupervised_loss < self.bestUnsupSum:
            self.bestUnsupSum = unsupervised_loss
            print("\n Best Target Unsup Loss so far: ", self.bestUnsupSum )
            self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, source_sup_target=1)


        if storeLoss:
            return float(running_loss / datasetSize)
    def evaluateUpdateLosses(self, model, testloader, targetloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=''):
        if unsupervised:
            #print('evaluating unsupervised performance')
            CAMLossInstance.cam_model.activations_and_grads.register_hooks()
            self.unsupervised_losses_source.append(self.evaluateModelUnsupervisedPerformance(model, testloader, CAMLossInstance, device, optimizer, storeLoss = True, batchDirectory= batchDirectory, source_or_target=0))
            self.unsupervised_losses_target.append(self.evaluateModelUnsupervisedPerformance(model, targetloader, CAMLossInstance, device, optimizer, storeLoss = True, batchDirectory= batchDirectory, source_or_target=1))
        #print('evaluating supervised performance')
        CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        self.evaluateModelSupervisedPerformance(model, testloader, criteron, device, optimizer, storeLoss = True, batchDirectory= batchDirectory, source_or_target=0)
        self.evaluateModelSupervisedPerformance(model, targetloader, criteron, device, optimizer, storeLoss = True, batchDirectory= batchDirectory, source_or_target=1)

        results = pd.DataFrame()        
        # results['Accuracy'] = self.accuracies       
        results['Supervised Loss Source'] = self.supervised_losses_source
        results['Supervised Loss Target'] = self.supervised_losses_target     
        results['Unsupervised Loss Source'] = self.unsupervised_losses_source  
        results['Unsupervised Loss Target'] = self.unsupervised_losses_target     
        results['accuracy_source'] = self.accuracy_source 
        results['accuracy_target'] = self.accuracy_target      
        results.to_csv(batchDirectory+'saved_figs/results.csv', header=True)
    def plotLosses(self, batchDirectory=''):
        plt.clf()
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].plot(self.supervised_losses_source, label="Supervised Loss Source")
        axs[0, 0].set_title('Supervised Loss Source')
        #plt.savefig(batchDirectory+'saved_figs/SupervisedLossPlot.png')
        #plt.clf()
        axs[0, 1].plot(self.unsupervised_losses_source, label="Unsupervised Loss Source")
        axs[0, 1].set_title('Unsupervised Loss Source')
        #plt.savefig(batchDirectory+'saved_figs/UnsupervisedLossPlot.png')
        #plt.clf()
        axs[0, 2].plot(self.accuracy_source, label="accuracy")
        axs[0, 2].set_title('accuracy source')
        #plt.savefig(batchDirectory+'saved_figs/TotalLossPlot.png')
        #plt.clf()

        axs[1, 0].plot(self.supervised_losses_target, label="Supervised Loss Target")
        axs[1, 0].set_title('Supervised Loss Target')

        axs[1, 1].plot(self.unsupervised_losses_target, label="Unsupervised Loss Target")
        axs[1, 1].set_title('Unsupervised Loss Target')

        axs[1, 2].plot(self.accuracy_target, label="accuracy")
        axs[1, 2].set_title('accuracy target')

        plt.tight_layout()
        plt.savefig(batchDirectory+'saved_figs/AllPlots.png')
        plt.close()
        # plt.legend()
    def calculateF1score(self, tp, fp, fn):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)
        
    def saveCheckpoint(self, net, optimizer, batchDirectory='',source_sup_target=1):
        if source_sup_target== 1:
            PATH = batchDirectory+"saved_checkpoints/model_best.pt"
        elif source_sup_target == 2:
            PATH = batchDirectory+"saved_checkpoints/model_best_accuracy_target.pt"
        elif source_sup_target == 0:
            PATH = batchDirectory+"saved_checkpoints/model_best_accuracy_source.pt"

        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)