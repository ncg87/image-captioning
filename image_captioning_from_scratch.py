import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
  
#CNN is encoder
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        #loading pretrained model for the encoding the images
        inception = models.inception_v3(weights='IMAGENET1K_V1', aux_logits = True)
        # making the parameters static, so we get same encodings for the same images
        for param in inception.parameters():
            param.requires_grad = False
        # take the inception model and remove the last 3 layers
        modules = list(inception.children())[:-3]
        # remove the auxillary output layer
        modules.remove(modules[15])
        # add relu layer to the end of the model
        self.inception = nn.Sequential(*modules,nn.ReLU())
        
        
    def forward(self, images):
        #running images through the inception model to get a features
        features = self.inception(images) # (batch_size, 2048, 8, 8)
        # change the shape of the features to (batch_size, 8, 8, 2048)
        features = features.permute(0, 2, 3, 1) # (batch_size, 8, 8, 2048)
        # smash the 8x8 pixels into a single dimension
        features = features.view(features.size(0), -1, features.size(-1))
        
        return features

# Attention mechanism for the decoder
class BasicAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BasicAttention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        
        self.A = nn.Linear(attention_dim, 1)
    def forward(self, feature, hidden_state):
        
        encoder_states = self.encoder_att(feature) # (batch_size, num_layers, attention_dim)
        decoder_states = self.decoder_att(hidden_state).unsqueeze(1) # (batch_size, attention_dim)
        
        combined_states = torch.tanh(encoder_states + decoder_states) # (batch_size, num_layers, attention_dim)
        
        # (batch_size, num_layers, attention_dim) -> (batch_size, num_layers, 1) -> (batch_size, num_layers)
        attention_scores = self.A(combined_states).squeeze(2)
        
        alpha = F.softmax(attention_scores, dim=1) # (batch_size, num_layers)
        
        attention_weights = feature * alpha.unsqueeze(2) # (batch_size, num_layers, features_dim) * (batch_size, num_layers, 1) = (batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1) # (batch_size, features_dim)
        
        return attention_weights
      

#RNN is decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_dim, attention_dim, vocab_size, p = 0.5):
        super(DecoderRNN, self).__init__()
        
        encoder_dim = 2048
        self.vocab_size = vocab_size
        
        
        self.embed = nn.Embedding(vocab_size, embed_size) 
        
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim) 
        
        self.attention = BasicAttention(encoder_dim, decoder_dim, attention_dim)
        
        self.linear = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p)
        
    def forward(self, features, captions):
        
        batch_size, seq_length  = captions.shape
        
        # vectorize the captions through embedding
        # (batch_size, captions_length) -> (batch_size, captions_length, embed_size)
        embeddings = self.dropout(self.embed(captions))
        # initialize the hidden and cell states of LSTM cell
        hidden, cell = self.init_hidden_state(features)
        # initalize tensor for predictions
        preds = torch.zeros(batch_size, seq_length, self.vocab_size)
        
        # one less so we are predicting the next token not the same token
        for t in range(seq_length-1):
            # get the attention weights
            attention_weights = self.attention(features, hidden)
            # concatenate the attention weights with the embeddings
            # (batch_size, features_dim + embed_size)
            input = torch.cat((attention_weights, embeddings[:, t]), dim=1)
            # pass the input to the LSTM cell
            hidden, cell = self.lstm(input, (hidden, cell))
            # get the predictions
            outputs = self.linear(self.dropout(hidden))
            preds[:, t] = outputs
        
        return preds
    
    def caption_image(self, features, vocabulary, max_length):
        
        # get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        caption = []
        
        # initalize hidden and cell with encoding features
        hidden, cell = self.init_hidden_state(features)
        # intialize start token , need to change the to cuda part
        first_word = torch.tensor(vocabulary.stoi['<SOS>']).view(1,-1).to(device)
        embedding = self.embed(first_word)
        # generate words until max_length or EOS token is generated
        for _ in range(max_length):
            # Get attention weights
            attention_weights = self.attention(features, hidden)
            # concatenate the attention weights with the embeddings
            input = torch.cat((attention_weights, embedding[:, 0]), dim=1)
            # get the hidden and cell states
            hidden, cell = self.lstm(input, (hidden, cell))
            # compute the predictions
            outputs = self.linear(self.dropout(hidden))
            # get the word with the highest probability
            predicted = outputs.argmax(1)
            caption.append(predicted.item())
            # end loop if EOS token is generated
            if vocabulary.itos[predicted.item()] == '<EOS>':
                break
            # update the embedding
            embedding = self.embed(predicted.unsqueeze(0))
            
        # convert the indices to words
        return [vocabulary.itos[idx] for idx in caption]
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_h(mean_encoder_out)  # (batch_size, encoder_dim)
        cell = self.init_c(mean_encoder_out) # (batch_size, encoder_dim)
        return hidden, cell
    
#combining the encoder to the decoder
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, decoder_dim, attention_dim, vocab_size, p = 0.5):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(embed_size, decoder_dim, attention_dim, vocab_size, p)
    
    #training
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    #inference
    def caption_image(self, image, vocabulary, max_length=50):
        
        with torch.inference_mode():
            # add batch dimension to the image
            features = self.encoderCNN(image)
            caption = self.decoderRNN.caption_image(features, vocabulary, max_length)

        return caption
                
        