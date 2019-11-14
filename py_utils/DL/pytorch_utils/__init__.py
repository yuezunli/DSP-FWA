"""
pytorch common utils
Written by Yuezun
"""

# torch.cuda.is_available() # Check whether cuda is available
# torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Set default tensor type

# ## Get Id of default device
# torch.cuda.current_device()
# # 0
# cuda.Device(0).name() # '0' is the id of your GPU
# # Tesla K80
#
# torch.cuda.get_device_name(0) # Get name device with ID '0'
# # 'Tesla K80'

# Releases all unoccupied cached memory currently held by
# the caching allocator so that those can be used in other
# GPU application and visible in nvidia-smi
# torch.cuda.empty_cache()

# if args.multigpu:
# net = torch.nn.DataParallel(net, device_ids=device_ids)
# net = net.cuda()
# cudnn.benckmark = True

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# Adam,.....

# if args.cuda:
# images = Variable(images.cuda())

# backprop
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# Save model
# torch.save({
#         'iter': i,
#         'lr': lr,
#         'state_dict': net.state_dict(),
#         'optimizer': optimizer.state_dict()},
#         model_path)


# Load model
# if os.path.isfile(model_path):
#     print("=> loading checkpoint '{}'".format(model_path))
#     checkpoint = torch.load(model_path)
#     start_iteration = checkpoint['iter']
#     lr = checkpoint['lr']
#     net.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("=> loaded checkpoint '{}' (iteration {})"
#           .format(model_path, checkpoint['iter']))