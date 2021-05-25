import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embedded_cap = self.embedding(captions[:,:-1])
        embedding = torch.cat((features.unsqueeze(1), embedded_cap), 1)
        x, _ = self.lstm(embedding)
        x = self.fc(x)
        
        return x
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        lstm_state = None
        for _ in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            output = self.fc(lstm_out)

            prediction = torch.argmax(output, dim=2)
            predicted_index = prediction.item()
            sentence.append(predicted_index)
            
            if predicted_index == 1:
                break
                
            inputs = self.embedding(prediction)

        return sentence