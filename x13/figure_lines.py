batch = 4
class_ = 9
plt.close('all')
plt.figure()
plt.imshow(rgb_f[batch,0].cpu())
plt.figure()
plt.imshow(gt_ss[batch,class_].cpu())
plt.figure()
plt.imshow(depth_f[batch,0].cpu())
plt.figure()
plt.imshow(top_view_sc[batch,class_].cpu())