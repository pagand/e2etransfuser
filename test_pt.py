import torch
import matplotlib.pyplot as plt

aa= torch.load('/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/depth.pt')

plt.figure()

plt.imshow(aa[0])