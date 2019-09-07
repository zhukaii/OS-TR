import numpy as np
# import matplotlib.pyplot as pyplot

np.random.seed(1)

# trans
def generate_collages(
        textures,
        segmentation_regions=5,
        anchor_points=None):
    N_textures = textures.shape[0]
    img_size = textures.shape
    masks = generate_random_masks(img_size, segmentation_regions, anchor_points)
    # mask:50*375*500*10
    batch = sum(textures[i] * masks[:, :, i:i+1] for i in range(segmentation_regions))
    return batch, masks[:, :, 0]


def generate_random_masks(img_size=(256, 256), segmentation_regions=5, points=None):
    xs, ys = np.meshgrid(np.arange(0, img_size[2]), np.arange(0, img_size[1]))

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1)
        points = np.random.randint(0, img_size[1], size=(n_points, 2))

    dists_b = [np.sqrt((xs - p[0]) ** 2 + (ys - p[1]) ** 2) for p in points]
    voronoi = np.argmin(dists_b, axis=0)
    masks_b = np.zeros((img_size[1], img_size[2], segmentation_regions), dtype=int)
    for m in range(segmentation_regions):
        masks_b[:, :, m][voronoi == m] = 1
    return masks_b

# def generate_validation_collages(N=240):
#     textures = np.load('./train_Onetf2.npy')
#    # textures = np.load('validation_textures_conv1_1.npy')
#    # print(textures.shape)
#    # pic = textures[1]
#    # pyplot.imshow(pic)
#     ids, mask, collages, point = generate_collages(textures, batch=N)
#     # print(ids[3][0], mask[0, :, :, 3])
#     len =collages.shape[0]
#     for i in range(0, len):
#         # print(point[i])
#         for j in range(5):
#             print(ids[j][i])
#         #     # ff = collages[1]
#         #     f = mask[i, :, :, j]
#         #     pyplot.figure(5*i+j)
#         #     pyplot.imshow(f)
#         #     pyplot.show()
#             # pyplot.figure(2)
#             # pyplot.imshow(f)
#    # np.save('validation_collages.npy', collages)
#
# if __name__ == '__main__':
#    generate_validation_collages()

# ijcai
# def generate_collages(
#         textures,
#         batch=1,
#         segmentation_regions=5,
#         anchor_points=None):
#     N_textures = textures.shape[0]
#     img_size = textures.shape
#     points, masks = generate_random_masks(img_size, batch, segmentation_regions, anchor_points)
#     # mask:50*375*500*10
#     textures_idx = [np.random.randint(0, N_textures, size=batch) for _ in range(segmentation_regions)]
#     batch = sum(textures[textures_idx[i]] * masks[:, :, :, i:i + 1] for i in range(segmentation_regions))
#     return textures_idx, masks, batch, points
#
#
# def generate_random_masks(img_size=(256, 256), batch=1, segmentation_regions=10, points=None):
#     xs, ys = np.meshgrid(np.arange(0, img_size[2]), np.arange(0, img_size[1]))
#
#     if points is None:
#         n_points = np.random.randint(2, segmentation_regions + 1, size=batch)
#         points = [np.random.randint(0, img_size[1], size=(n_points[i], 2)) for i in range(batch)]
#
#     masks = []
#     for b in range(batch):
#         dists_b = [np.sqrt((xs - p[0]) ** 2 + (ys - p[1]) ** 2) for p in points[b]]
#         voronoi = np.argmin(dists_b, axis=0)
#         masks_b = np.zeros((img_size[1], img_size[2], segmentation_regions), dtype=int)
#         for m in range(segmentation_regions):
#             masks_b[:, :, m][voronoi == m] = 1
#         masks.append(masks_b)
#     return n_points, np.stack(masks)