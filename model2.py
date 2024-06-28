import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# Encoder Block
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
        # running images through the inception model to get a features
        features = self.inception(images) #(batch_size,2048,7,7)
        # change the shape of the features to (batch_size, 8, 8, 2048)
        features = features.permute(0, 2, 3, 1)
        # smash the 8x8 pixels into a single dimension
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        
        return features
# Basic attention mechanism for decoder
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        # Initialize attention dimensions
        self.attention_dim = attention_dim
        # Linear transformation for decoder hidden state
        self.W = nn.Linear(decoder_dim,attention_dim)
        # Linear transformation for encoder features
        self.U = nn.Linear(encoder_dim,attention_dim)
        # Linear transformation to scalar for attention scores
        self.A = nn.Linear(attention_dim,1)
         
    def forward(self, features, hidden_state):
        # Transform encoder hidden states (features)
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        # Transform decoder hidden state
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        # Combine encoder and decoder hidden states with tanh activation
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        # Calculate attention scores using linear transformation
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        # Remove the last dimension (scalar) from attention scores
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        # Apply softmax to calculate attention weights
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        # Compute attention weights by multiplying features with alpha (broadcasting)
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        # Sum along num_layers dimension to get final attention weights
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return attention_weights

#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, features, captions):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # vectorize the caption
        embeds = self.drop(self.embedding(captions))
        
        # Initialize LSTM hidden and cell states
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0) # get batch size to create tensor below
        # initalize tensor for prediction
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        # Predict tokens     
        for s in range(seq_length):
            # get the attention weights
            context = self.attention(features, h)
            # concatenate the attention weights with the embeddings
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            # pass the input to the LSTM cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            # get the predictions       
            output = self.fcn(self.drop(h))
            preds[:,s] = output
        
        return preds
    
    def generate_caption(self,features,max_len=20,vocab=None):
        # get device to initialize tensor below to correct device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Given the image features generate the captions
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        # initialize starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)
        # initalize word array for captions
        captions = []
        
        for i in range(max_len):
            # Get attention weights
            context = self.attention(features, h)
            
            # concat embed and attention weights
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            # get word probs output
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            #select the word with highest prob
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            #send predicted word as the next word to predict on
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions]
    
    # initializing the hidden and cell with a linear transformation of the mean of the features
    def init_hidden_state(self, encoder_out):
        # get mean of features across the layers dimension
        # (batch_size, num_layers, decoder_dim) -> (batch_size, decoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1)
        # linear transformation of the encoder features mean
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, decoder_dim, attention_dim, vocab_size, p=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=2048,
            decoder_dim=decoder_dim,
            drop_prob = p
        )
        
    def forward(self, images, captions):
        # run encoder and decoder
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def caption_image(self, image, vocab, max_len=20):
        
        with torch.inference_mode():
                # case of single image input
                if(image.dim() == 3):
                    image.unsqueeze(0)
                # get features of image
                features = self.encoder(image)
                # generate caption for image
                caption = self.decoder.generate_caption(features, max_len, vocab)
                
        return caption
                