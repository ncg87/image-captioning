# Import statements
import os # For loading file paths
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  # Pad batches
from torchvision import transforms
import spacy # For tokenizer
import pandas as pd # For reading annotation file
from PIL import Image # Load images

# Load spacy english NLP model
spacy_eng = spacy.load('en_core_web_sm')

# Class to build vocabulary
class Vocabulary:
    """Builds vocabulary for a given input of sentences"""
    def __init__(self, freq_threshold):
        # Base tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        # Frequency threshold, otherwise <UNK>
        self.freq_threshold = freq_threshold
    
    # Returns length of vocabulary
    def __len__(self):
        return len(self.itos)

    # Tokenizes a sentence / input
    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        # Dictionary to keep track of frequencies of words
        frequencies = {}
        # Starting index of first new word (since four words are already contained)
        idx = 4
         # Iterates through all sentences in list
        for sentence in sentence_list:
            # Iterates through all words in given sentence
            for word in self.tokenizer_eng(sentence):
                # If not in dictionary add to dictionary as seen once
                if word not in frequencies:
                    frequencies[word] = 1
                # If seen increase frequency
                else:
                    frequencies[word] +=1
                # If frequency reaches threshold add to vocab
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx +=1
    
    def numericalize(self, text):
        # Tokenize input text / sentence
        tokenized_text = self.tokenizer_eng(text)

        # Numericalize tokens of sentence
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# Base transform
base_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

# Dataset class, initialization of the dataset
class Flickr8kDataset(Dataset):
    """Flickr 8k Image Captioning Dataset"""
    def __init__(self, root_path, annotation_file, transform = base_transform, freq_threshold = 4):
        
        # Path of images
        self.root_path = root_path
        # Transformation for images
        self.transform = transform
        # Dataframe of annotations
        self.df = pd.read_csv(annotation_file)
        
        # Image column
        self.images = self.df["image"]
        # Caption column
        self.captions = self.df["caption"]
        
        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __getitem__(self, index):
        # Get image id and caption
        img_id = self.images[index]
        caption = self.captions[index]
        # Get image in RGB
        img = Image.open(os.path.join(self.root_path, img_id)).convert("RGB")
        
        # Transform image
        img = self.transform(img)
        
        # Start numericalization with SOS token
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        # Numericalize caption and merge lists
        numericalized_caption += self.vocab.numericalize(caption)
        # Numericalize EOS token and append to end
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        caption = torch.tensor(numericalized_caption)
        
        return img, caption
    # Returns length of dataset
    def __len__(self):
        return len(self.images)

# Padding class for dataloader
class Collate:
    # Initialize idx for padding
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    # Call function
    def __call__(self, batch):
        # Get list of images
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        # Get list of all numericalize captions
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value = self.pad_idx)
        
        return imgs, targets
