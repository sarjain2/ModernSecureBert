import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaModel

device = "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

# Load fine-tuned weights
checkpoint = torch.load("/Users/sarjain2/Downloads/checkpoint_epoch_20.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set model to evaluation mode

tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = AutoModelForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")
model.eval()

# Input text with a [MASK] token
inputs = "Search order attacks [MASK] when an adversary abuses the order in which Windows searches for programs that are not given a path"
inputs = ''' When an application loads a DLL without specifying the full path, Windows searches directories in a specific order. This can be exploited by an attacker, as shown below:
 
python
Copy Code
import ctypes
import os

# Step 1: Simulate a malicious DLL in an <mask> <mask>
os.makedirs("attacker_dir", exist_ok=True)
with open("attacker_dir/malicious.dll", "w") as f:
    f.write("Malicious code here")

# Step 2: Add the attacker's directory to the PATH environment variable
os.environ["PATH"] = "attacker_dir;" + os.environ["PATH"]

# Step 3: Vulnerable behavior - Load the DLL without specifying the full path
try:
    ctypes.CDLL("malicious.dll")  # Loads the malicious DLL
    print("Malicious DLL loaded!")
except Exception as e:
    print(f"Error loading DLL: {e}") '''
    
inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True).to(device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Find the predicted token for [MASK]
masked_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
for id in predicted_token_id:
    predicted_token = tokenizer.decode(id.item())
    print(f"The predicted token is: {predicted_token}")