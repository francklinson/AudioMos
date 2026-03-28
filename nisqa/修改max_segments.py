import torch

checkpoint = torch.load('weights/nisqa.tar', map_location=torch.device('cpu'))
checkpoint['args']['ms_max_segments'] = 3000
torch.save(checkpoint, 'weights/nisqa_3000.tar')
