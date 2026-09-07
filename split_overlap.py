import numpy as np

def split_3d_image_overlap(image, block_size, overlap):

    shape = image.shape
    if len(shape) != 3:
        raise ValueError("The input image must be 3D")

    num_blocks = [(d + block_size - 1) // block_size for d in shape]
    blocks = []

    for iz in range(num_blocks[0]):
        z_start = iz * block_size
        z_end = min(z_start + block_size + overlap, shape[0])
        for iy in range(num_blocks[1]):
            y_start = iy * block_size
            y_end = min(y_start + block_size + overlap, shape[1])
            for ix in range(num_blocks[2]):
                x_start = ix * block_size
                x_end = min(x_start + block_size + overlap, shape[2])
                block = image[z_start:z_end, y_start:y_end, x_start:x_end]
                blocks.append(block)

    return blocks, shape, block_size, overlap, num_blocks


def _get_1d_weight(length, idx, num_blocks, overlap):
    w = np.ones(length, dtype=np.float64)
    if idx > 0:
        left = min(overlap, length)
        if left > 0:
            w[:left] = np.linspace(0, 1, left)
    if idx < num_blocks - 1:
        right = min(overlap, length)
        if right > 0:
            w[-right:] = np.linspace(1, 0, right)
    return w


def combine_3d_blocks_overlap(blocks, original_shape, block_size, overlap, num_blocks):
    accum = np.zeros(original_shape, dtype=np.float64)
    weight_sum = np.zeros(original_shape, dtype=np.float64)

    idx = 0
    for iz in range(num_blocks[0]):
        z_start = iz * block_size
        z_end = min(z_start + block_size + overlap, original_shape[0])
        Lz = z_end - z_start
        w_z = _get_1d_weight(Lz, iz, num_blocks[0], overlap)

        for iy in range(num_blocks[1]):
            y_start = iy * block_size
            y_end = min(y_start + block_size + overlap, original_shape[1])
            Ly = y_end - y_start
            w_y = _get_1d_weight(Ly, iy, num_blocks[1], overlap)

            for ix in range(num_blocks[2]):
                x_start = ix * block_size
                x_end = min(x_start + block_size + overlap, original_shape[2])
                Lx = x_end - x_start
                w_x = _get_1d_weight(Lx, ix, num_blocks[2], overlap)

                weight_3d = w_z[:, None, None] * w_y[None, :, None] * w_x[None, None, :]

                block = blocks[idx].astype(np.float64)
                accum[z_start:z_end, y_start:y_end, x_start:x_end] += block * weight_3d
                weight_sum[z_start:z_end, y_start:y_end, x_start:x_end] += weight_3d
                idx += 1

    result = accum / (weight_sum + 1e-12)

    if blocks[0].dtype.kind in 'iu':
        result = np.rint(result).astype(blocks[0].dtype)
    else:
        result = result.astype(blocks[0].dtype)

    return result