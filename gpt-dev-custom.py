"""
 what is the functionality we are trying to acheive?
    -> Text generation

 given a set of text, we want to generate the next text
 
 The generation of the next has to be determined from the
 characteristics of the text preceeding it.

 pretty simple,
 So the information that is contained in the preceeding text determines the most 
 probably next set of text

"""
import torch
device= "mps" if torch.backends.mps.is_available() else "cpu"

if torch.backends.mps.is_built():
    print("MPS is built")
else:
    print("MPS is not built")


# read the input file
with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# check the input file is being read correctly
# print(text[:100])

# now we need to construct a vocabulary
# here we have choices to construct vocab of various sizes:
# lets go ahead with character level vocab for now

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
"""
print(''.join(chars))
print(chars)
print(vocab_size) # outputs 65
"""

# lets also create a encoding and decoding function to represent out tokens
# so each character will be represented by a number
# that number in itself does not carry information, its just to indentify it

# what does encode do?
# it takes a character and outputs a number
# lets create a dict that maps characters to their index and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
# print(stoi)
itos = { i:ch for i,ch in enumerate(chars) }
# print(itos)
# what does decoder do?
# takes a number, outputs a character
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# create a mapping from characters to integers

# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

# these encodings will be fed to the model
# And a set of encodings will be fed
# for example, [token1, token2, token3 ... token_N] will represent a sequence
# where each token is a number or index

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# splitting data into train and validation set
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# let's define block_size
# here the block size refers to the max size of context we are processing
block_size = 8
# block_size -> input, +1 -> becomes the output
# print(train_data[:block_size+1]) # example sequence

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
# here the batch size is different than block_size
# batches are used for efficiency with GPU

# lets create a get_batch to get train and valid batches
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # choose between train or split data
    data = train_data if split == 'train' else val_data

    # get start indexes. #ix = batch_size (4) -> so we take 4
    # get batch_size random indices
    start_ix = torch.randint(len(data) - block_size, (batch_size,))

    # get x input
    x = torch.stack([data[i:i+block_size] for i in start_ix])
    # get y input, it being +1 indexed from the input x 
    y = torch.stack([data[i+1:i+block_size+1] for i in start_ix])
    x,y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
# print(xb)
print('targets:')
print(yb.shape)
# print(yb)




from models import BigramLanguageModel
# print(torch.__file__)
m = BigramLanguageModel(vocab_size).to(device) # this feeds to the model initialization 
logits, loss = m(xb, yb) # this feeds to the forward function
print('logits shape: ', logits.shape)
print('Loss: ', loss)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("Model parameters:")
for name, param in m.named_parameters():
    if param.requires_grad:
        print(f"  {name}: shape={param.shape}")


print(decode(m.generate(max_new_tokens=20, xb=context)[0].tolist()))
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)[0].tolist()))


# lets train the model now
# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(max_new_tokens=100, xb=context)[0].tolist()))