import torch
from torch import nn
from torch.utils import DataLoader, Subset
import json, os, time
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from ViTransformer import ViT, DINOHead
import utils
import pathlib
import tqdm
from evaluate import compute_embedding, compute_knn
from utils import DataHandler
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
    
TRAIN_INPUT = config['DATA']['TRAIN_INPUT']
TEST_INPUT = config['DATA']['TEST_INPUT']
CLASSES = config['DATA']['CLASSES']
PATCH_SIZE = int(config['PARAMETERS']['patch_size'])
HIDDEN_SIZE = int(config['PARAMETERS']['hidden_size'])
NUM_ATTENTION_HEADS = int(config['PARAMETERS']['num_attention_heads'])
NUM_CHANNELS = int(config['PARAMETERS']['num_channels'])
NUM_CLASSES = int(config['PARAMETERS']['num_classes'])
INTERMEDIATE_SIZE = int(config['PARAMETERS']['intermediate_size'])
IMAGE_SIZE = int(config['PARAMETERS']['image_size'])
EXP_NAME = config['PARAMETERS']['exp_name']
BATCH_SIZE = int(config['PARAMETERS']['batch_size'])
EPOCHS = int(config['PARAMETERS']['epochs'])
LR = float(config['PARAMETERS']['lr'])
SAVE_MODEL_EVERY = int(config['PARAMETERS']['save_model_every'])
DATASET =  config['PARAMETERS']['dataset_name']
OUT_DIM = config['PARAMETERS']['out_dim']
# Make the output satisfying
class TextColors:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    ENDC = "\033[0m"

        

        
        
    



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using the {TextColors.PURPLE}{torch.cuda.get_device_name(0)}{TextColors.ENDC} for training.")
# These are not hard constraints, but are used to prevent misconfigurations
assert HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0, "The size of hidden layer is not divisible by the number of attention heads"
assert INTERMEDIATE_SIZE == 4 * HIDDEN_SIZE, "The intermediate size is not 4 times the embedding/hidden size."
assert IMAGE_SIZE % PATCH_SIZE == 0, "Image size is not divisible by patch size"



class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, student_model, teacher_model, optimizer, classifier, loss_fn, exp_name, device):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.optimizer = optimizer
        self.classifier = classifier.to(device)
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        
        
    def pretrain(self, pretrainloader, epochs, save_model_every_n_epochs=0):
        """
        Pre-train the model for the specified number of epochs.
        """
        for epoch in range(epochs):
            pretrain_loss, pretrain_accuracy = self.train_epoch(pretrainloader)
            print(f"Pretrain Epoch: {epoch+1}, Loss: {pretrain_loss:.4f}, Accuracy: {pretrain_accuracy:.4f}")
            

    

    def finetune(self, finetuneloader, epochs, save_model_every_n_epochs=0):
        """
        Fine-tune the model for the specified number of epochs.
        """
        for epoch in range(epochs):
            finetune_loss, finetune_accuracy = self.train_epoch(finetuneloader)
            print(f"Fine-tune Epoch: {epoch+1}, Loss: {finetune_loss:.4f}, Accuracy: {finetune_accuracy:.4f}")

    
        
    def train(self, 
              trainloader,
              valloader, 
              augloader, 
              epochs, 
              dataset_train_aug, 
              student, 
              teacher, 
              data_loader_val_plain_subset,
              optimizer,
              loss_fn,
              teacher_momentum,
              student_momentum,
              ):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        t0 = time.time()
        n_steps = 0
        best_acc = 0
        num_batches = len(dataset_train_aug) // BATCH_SIZE
        # Train the model
        for i in range(epochs):
            for j, (images, _) in tqdm.tqdm(enumerate(augloader), total=num_batches):
                student.eval()
                embs, imgs, labels_ = compute_embedding(
                    student.backbone,
                    data_loader_val_plain_subset
                )

                
                curr_acc = compute_knn(
                    student.backbone,
                    trainloader,
                    valloader,
                )
                if curr_acc > best_acc:
                    best_acc = curr_acc
                
                student.train()
            
                images = [img.to(device) for img in images]
                student_logits = student(images)
                teacher_logits = teacher(images[:2])
                loss = loss_fn(student_logits, teacher_logits)
                optimizer.zero_grad()
                loss.backward()
                utils.clip_gradients(student)
                optimizer.step()

                with torch.no_grad():
                    for student_ps, teacher_ps in zip(
                        student.parameters(), teacher.parameters()
                    ):
                        teacher_ps.data.mul_(teacher_momentum)
                        teacher_ps.data.add_(
                            (1 - teacher_momentum) * (student_ps.detach()).data
                        )
                n_steps += 1
                
            # train_loss, train_accuracy = self.train_epoch(trainloader)
            # test_accuracy, test_loss = self.evaluate(testloader)
            # train_losses.append(train_loss)
            # test_losses.append(test_loss)
            # accuracies.append(train_accuracy)
            t1 = time.time()
            print(f"{TextColors.LIGHT_BLUE}Epoch:{TextColors.ENDC} {i+1}") 
            print(f"{TextColors.BLUE}Train loss:{TextColors.ENDC} {train_loss:.4f}")
            print(f"{TextColors.CYAN}Test loss:{TextColors.ENDC} {test_loss:.4f}") 
            print(f"{TextColors.GREEN}Accuracy:{TextColors.ENDC} {test_accuracy:.4f}\n")
            print(f"Time Elapsed from Last Epoch: {t1 - t0}")
            t0 = t1
            #TODO ADD VISUALIZATION STEP HERE
        

    # def train_epoch(self, trainloader):
    #     """
    #     Train the model for one epoch.
    #     """
    #     self.student_model.train()
    #     self.classifier.train()
    #     total_loss = 0
    #     correct = 0
    #     for batch in trainloader:
    #         # Move the batch to the device
    #         batch = [t.to(self.device) for t in batch]
    #         images, labels = batch
    #         # Zero the gradients
    #         self.optimizer.zero_grad()
    #         # Calculate the loss
    #         # logits = self.classifier(self.student_model(images))
    #         student_logits = self.student_model(images)
    #         teacher_logits = self.teacher_model(images[:2]) #TODO ????
    #         # logits = self.classifier(self.model.get_intermediate_layers(images))
    #         loss = self.loss_fn(student_logits, teacher_logits)
    #         # Backpropagate the loss
    #         loss.backward()
    #         # Update the model's parameters
    #         self.optimizer.step()
    #         total_loss += loss.item() * len(images)
    #         # Calculate the number of correct predictions
    #         predictions = torch.argmax(logits, dim=1)
    #         correct += torch.sum(predictions == labels).item()
    #     accuracy = correct / len(trainloader.dataset)
    #     return total_loss / len(trainloader.dataset), accuracy

    @torch.no_grad()
    def evaluate(self, testloader):
        self.student_model.eval()
        self.classifier.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            #go through entire batch
            for batch in testloader:
                #send each to GPU
                batch = [t.to(self.device) for t in batch]
                # Get predictions
                images, labels = batch
                
                features  = self.student_model(images)
                logits = self.classifier(features)
                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)
                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss

# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""
#     def __init__(self, dim, num_labels=NUM_CLASSES):
#         super(LinearClassifier, self).__init__()
#         self.num_labels = num_labels
#         self.linear = nn.Linear(dim, num_labels)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()

#     def forward(self, x):
#         # flatten
#         x = x.view(x.size(0), -1)

#         # linear layer
#         return self.linear(x)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, x):
        return self.linear(x)

def main():
    # Training parameters
    save_model_every_n_epochs = SAVE_MODEL_EVERY
    # Load the dataset
    
    path_dataset_train = pathlib.Path("data/data_imagenette/train")
    path_dataset_val = pathlib.Path("data/data_imagenette/val")
    classes_path = pathlib.Path('data/data_imagnette/imagenette_labels.json')
    
    with classes_path.open('r') as f:
        classes = json.load(f)
    
    
    
    
    
    
    transform_aug = utils.DataAugmentation(size=224, n_local_crops = 2)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_aug = nn.ImageFolder(path_dataset_train, transform=transform_aug)
    dataset_train_plain = nn.ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = nn.ImageFolder(path_dataset_val, transform=transform_plain)
    
    aug_loader = DataLoader(
        dataset_train_aug,
        batch_size= 32,
        shuffle=True,
        drop_last=True,
        num_workers= 4,
        pin_memory=True,
    )
    train_loader = DataLoader(
        dataset_train_plain,
        batch_size= 32,
        drop_last= False,
        num_workers= 4,
    )
    val_loader = DataLoader(
        dataset_val_plain,
        batch_size = 32,
        drop_last = False,
        num_workers = 4,
    )
    
    val_subset_loader = DataLoader( #TODO find out what this does
        dataset_val_plain,
        batch_size= 32,
        drop_last = False,
        sampler = nn.utils.SubsetRandomSampler(list(range(0, len(dataset_val_plain), 50))),
        num_workers = 4,
    )

   
    
    # Create the model, optimizer, loss function and trainer
    student_model = ViT(img_size = IMAGE_SIZE,
                patch_size= PATCH_SIZE,
                in_chans= NUM_CHANNELS,
                num_classes= NUM_CLASSES,
                embed_dim= HIDDEN_SIZE)
    teacher_model = ViT(img_size = IMAGE_SIZE,
                patch_size= PATCH_SIZE,
                in_chans= NUM_CHANNELS,
                num_classes= NUM_CLASSES,
                embed_dim= HIDDEN_SIZE)
    for p in teacher_model.parameters():
        p.requires_grad = False
        
    student = utils.MultiCropWrapper(student_model, DINOHead(
        HIDDEN_SIZE,
        OUT_DIM     
    ))
    teacher = utils.MultiCropWrapper(teacher_model, DINOHead(
        HIDDEN_SIZE,
        OUT_DIM     
    ))
    student.cuda()
    teacher.cuda()
    for p in teacher.parameters():
        p.requires_grad = False
    
    dino_loss = utils.Loss(
        out_dim=OUT_DIM,
        teacher_temp = 0.04, #TODO turn to configs
        student_temp=0.1
    ).cuda()
    # classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    classifier = LinearClassifier(HIDDEN_SIZE, num_labels = NUM_CLASSES)
    optimizer = optim.AdamW(list(student_model.parameters()) + list(classifier.parameters()), lr=LR, weight_decay=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay = 1e-2, momentum = 0.9)
    # loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(student_model, teacher_model, optimizer, classifier, dino_loss, EXP_NAME, device=device)
    print("==============MIND DINO Phase==============")
    # trainer.pretrain(pretrainloader, PRETRAIN_EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Training MIND==============")
    trainer.train(trainloader, testloader, EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Validating MIND==============")
    # trainer.finetune(finetuneloader, FINETUNE_EPOCHS, save_model_every_n_epochs=save_model_every_n_epochs)
    print("==============Training done==============")

if __name__ == '__main__':
    main()