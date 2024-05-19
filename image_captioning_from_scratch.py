import torch
import torch.nn as nn
import torchvision.models as models

#CNN is encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        #loading pretrained model fpr the first part of the model, encodings
        #no aux logits is the checkpoint results output from each layer
        self.inception = models.inception_v3(weights='IMAGENET1K_V1', aux_logits = True)
        #taking last layer of inception model/ fc, and mapping it to a new head, which will be our RNN
        #pretty much just changing the last layer of the inception model to fit our RNN, a vector of our embedding size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        #running images through the inception model
        features = self.inception(images)
        
        #fine tuning the last layer, by only turning on grads for the last layer
        for name, param in self.inception.named_parameters():
            #specificly the weights and biases of the neurons in the last layer
            #the fc layer that was defined above
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad =True
            #for if train, the initialization is set to true, the case that you want to train the whole model
            #could probably use this exact method to fine tune further down the network if wanted to
            else:
                param.requires_grad = self.train_CNN
        return self.dropout(self.relu(features[0]))
    
#RNN is decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
#combining the encoder to the decoder
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    #training
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    #inference
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            #getting indices for caption and inputing each one for predict next word
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(0)
                
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
        # return an array of the string to caption
        return[vocabulary.itos[idx] for idx in result_caption]
                
        