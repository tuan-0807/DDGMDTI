import torch
import copy
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, \
    precision_score
from model import binary_cross_entropy, cross_entropy_logits
from tqdm import tqdm



class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader,
                 hyperparam_dict, alpha=1):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = hyperparam_dict['MAX_EPOCH']  # 100
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = hyperparam_dict['DECODER_BINARY']  # 1

        self.nb_training = len(self.train_dataloader)  # 训练集的数量/64
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0



        # 添加特征文件路径模板
        self.prott5_template = {
            'train': '../ProtT5_embedding/biosnap/train/train_pro_emd_{}.npy',
            'val': '../ProtT5_embedding/biosnap/val/val_pro_emd_{}.npy',
            'test': '../ProtT5_embedding/biosnap/test/test_pro_emd_{}.npy'
        }
        self.mol_template = {
            'train': '../MolFormer_embedding/biosnap/seed/train/train_drug_emd_{}.npy',
            'val': '../MolFormer_embedding/biosnap/seed/val/val_drug_emd_{}.npy',
            'test': '../MolFormer_embedding/biosnap/seed/test/test_drug_emd_{}.npy'
        }
        self.dismap_template = {
            'train': '../dismap_adj/biosnap/train/train_pro_adj_{}.npy',
            'val': '../dismap_adj/biosnap/val/val_pro_adj_{}.npy',
            'test': '../dismap_adj/biosnap/test/test_pro_adj_{}.npy'
        }

    def load_features(self, data_type, batch_idx):
        """按需加载特征文件"""
        prott5_file = self.prott5_template[data_type].format(batch_idx + 1)
        mol_file = self.mol_template[data_type].format(batch_idx + 1)
        dismap_file = self.dismap_template[data_type].format(batch_idx + 1)

        prott5_emd = np.load(prott5_file)
        mol_emd = np.load(mol_file)
        dismap = np.load(dismap_file)[:-1, :-1]  # 应用切片

        return (
            torch.tensor(prott5_emd).to(self.device),
            torch.tensor(mol_emd).to(self.device),
            torch.tensor(dismap).to(self.device)
        )


    def train(self):

        # 加载预训练模型提取的 训练集、验证集和测试集的特征
        #prott5_train_list, prott5_val_list, prott5_test_list, mol_train_list, mol_val_list, mol_test_list,dismap_train_list,dismap_val_list,dismap_test_list= (
        #    loadProtT5andMol(len(self.train_dataloader), len(self.val_dataloader), len(self.test_dataloader)))

        print("load...1")
        best_val_loss = float('inf')  # 初始化最佳验证损失


        for i in range(self.epochs):  # self.epochs = 100
            self.current_epoch += 1  # current_epoch初始值为0

            # 训练
            train_loss = self.train_epoch()

            # 下面进行验证
            #auroc, auprc, val_loss = self.test(prott5_val_list, prott5_test_list,
            #                                   mol_val_list, mol_test_list,
            #                                   dismap_val_list,dismap_test_list,
            #                                   dataloader="val")
            auroc, auprc, val_loss = self.test(dataloader="val")
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                torch.save(self.model.state_dict(), '../model_result/biosnap/model_best_params.pth')
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))

        # 100个epoch后，进行测试!
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = (
            #self.test(prott5_val_list, prott5_test_list,
            #          mol_val_list, mol_test_list,
            #          dismap_val_list, dismap_test_list,
            #          dataloader="test"))
             self.test(dataloader="test"))
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " F1 " + str(f1) +" Thred_optim " + str(thred_optim))



    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)  # 训练集的样本数19224/64 = 301

        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):  # 会自动调用DTIDataset类里面的__getitem__方法
            self.step += 1
            # 按需加载当前batch的特征
            train_prott5_emd, train_mol_emd, train_dismap = self.load_features('train', i)

            #train_prott5_emd = (torch.tensor(prott5_train_list[i])).to(self.device)
            #train_mol_emd = (torch.tensor(mol_train_list[i])).to(self.device)
            #train_dismap=(torch.tensor(dismap_train_list[i])).to(self.device)

            labels = labels.float().to(self.device)


            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(train_mol_emd, train_prott5_emd,train_dismap)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

            del train_prott5_emd, train_mol_emd, train_dismap, v_d, v_p, f, score, n, loss
            torch.cuda.empty_cache()

        loss_epoch = loss_epoch / num_batches  # 这里输出的loss是某一轮每个batch的loss的平均值
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch


    def test(self,dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                labels = labels.float().to(self.device)
                if dataloader == "val":
                    val_prott5_emd, val_mol_emd, val_dismap = self.load_features('val', i)
                    #val_prott5_emd = (torch.tensor(prott5_val_list[i])).to(self.device)
                    #val_mol_emd = (torch.tensor(mol_val_list[i])).to(self.device)
                    #val_dismap=(torch.tensor(dismap_val_list[i])).to(self.device)

                    v_d, v_p, f, score = self.model(val_mol_emd, val_prott5_emd,val_dismap)

                elif dataloader == "test":
                    test_prott5_emd, test_mol_emd, test_dismap = self.load_features('test', i)
                    #test_prott5_emd = (torch.tensor(prott5_test_list[i])).to(self.device)
                    #test_mol_emd = (torch.tensor(mol_test_list[i])).to(self.device)
                    #test_dismap=(torch.tensor(dismap_test_list[i])).to(self.device)


                    v_d, v_p, f, score = self.best_model(test_mol_emd, test_prott5_emd,test_dismap)

                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()  # 所有batch合并在一起的真实值
                y_pred = y_pred + n.to("cpu").tolist()  # 所有batch合并在一起的真实值
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
