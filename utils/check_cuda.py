import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    num_gpu = int(input(f"Enter number of gpu from ({[i for i in range(num_gpus)]}):  "))
    device = f'cuda:{num_gpu}'

else:
    print("CUDA is not available.\n Good job , now go on cpu :)")
    device = 'cpu'



#  ----------- ML code  ---------------
# Moving a tensor to a specific GPU:
tensor = torch.randn(3)               # Move a tensor to GPU 0
tensor = tensor.to(device)            # tensor = tensor.cuda(num_gpu)       # Alternatively, you can use .cuda()

# Moving a model to a specific GPU:
model = torch.nn.Linear(3,1)
model = model.to(device)               # Move the model to GPU 0      #model = model.cuda(num_gpu)        # Alternatively, you can use .cuda()

# Set the default GPU:
torch.cuda.set_device(num_gpu)
print(model(tensor).detach())
# ------------------------------------------


