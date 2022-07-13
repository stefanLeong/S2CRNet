import torch

def BGR_to_RGB(image):
    "Just permutes the color channels"
    if len(image.size())==3:
        return image[2,1,0]
    elif len(image.size())==4:
        return image[:,[2,1,0]]

def RGB_to_BGR(image):
    "Just permutes the color channels"
    return BGR_to_RGB(image)

def srgb_to_rgb(image):
    """Linearizes sRGB to RGB. Assumes input is in range [0,1].
    Works for batched images too.


    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are gamma-corrected R, G, B
    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are linearized RGB
    """
#     assert torch.max(image) <= 1

    small_u = image/12.92
    big_u = torch.pow((image+0.055)/1.055, 12./5)

    return torch.where(image<=0.04045, small_u, big_u)

def rgb_to_srgb(image):
    """Applies gamma correction to rgb to get sRGB. Assumes input is in range [0,1]
    Works for batched images too.

    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are linearized R, G, B
    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are gamma-corrected RGB
    """
#     assert torch.max(image) <= 1

    small_u = image*12.92
    big_u = torch.pow(image,.416)*1.055-0.055

    return torch.where(image<=0.0031308, small_u, big_u)


def rgb_to_xyz(images):
    """
    Converts true (linearized) rbg to xyz.
        Works for batched images too.


    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are true R, G, B
                        OR a batched version with (N, 3, x, y)

    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are X, Y, Z
                    Or the batched version

    """
    M65 = torch.tensor([[0.4124564,  0.3575761,  0.1804375],
                         [0.2126729,  0.7151522,  0.0721750],
                         [0.0193339,  0.1191920 , 0.9503041]]).to(images.device)

                    # multiply by the matrix only along the color dimension
    images = torch.einsum("...jkl,jm->...mkl",images, M65)

    return images

def xyz_to_rgb(images):
    """
    Converts xyz to true (linearized) rbg.
        Works for batched images too.


    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are true X, Y, Z
                    OR a batched version with (N, 3, x, y)
    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are R, G, B
                Or the batched version
    """
    M65 = torch.tensor([[0.4124564,  0.3575761,  0.1804375],
                         [0.2126729,  0.7151522,  0.0721750],
                         [0.0193339,  0.1191920 , 0.9503041]]).to(images.device)
    M65_inv = torch.inverse(M65)

    # multiply by the matrix only along the color dimension. Wor
    images = torch.einsum("...jkl,jm->...mkl",images, M65_inv)

    return images

def luv_to_xyz(image):
    """
    Converts luv to xyz. Assumes D65 standard illuminant.

    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are X, Y, Z
    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are L, U, V
    """
    # single image
    if len(image.size()) == 3:
        u_prime = image[1] / (13 * image[0]) + .2009
        v_prime = image[2] / (13 * image[0]) + .4610

        small_Y = image[0] * (3. / 29) ** 3
        large_Y = ((image[0] + 16.) / 116.) ** 3

        Y = torch.where(image[0] <= 8, small_Y, large_Y)
        d = 0
        # batch of images
    elif len(image.size()) == 4:

        u_prime = image[:, 1] / (13 * image[:, 0]) + .2009
        v_prime = image[:, 2] / (13 * image[:, 0]) + .4610

        small_Y = image[:, 0] * (3. / 29) ** 3
        large_Y = ((image[:, 0] + 16.) / 116.) ** 3

        Y = torch.where(image[:, 0] <= 8, small_Y, large_Y)
        d = 1

    X = Y * 9 * u_prime / (4 * v_prime)
    Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

    xyz_image = torch.stack((X, Y, Z), dim=d)

    return xyz_image


def xyz_to_luv(image):
    """
    Converts xyz to luv. Assumes D65 standard illuminant.

    :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are XYZ
    :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are LUV
    """
    # single image
    if len(image.size()) == 3:
        small_L = (29. / 3) ** 3 * image[1]
        large_L = 116 * torch.pow(image[1], 1 / 3.) - 16
        L = torch.where(image[1] <= (6. / 29) ** 3, small_L, large_L)

        u_prime = 4 * image[0] / (image[0] + 15 * image[1] + 3 * image[2])
        v_prime = 9 * image[1] / (image[0] + 15 * image[1] + 3 * image[2])
        d = 0
    elif len(image.size()) == 4:
        small_L = (29. / 3) ** 3 * image[:, 1]
        large_L = 116 * torch.pow(image[:, 1], 1 / 3.) - 16
        L = torch.where(image[:, 1] <= (6. / 29) ** 3, small_L, large_L)

        u_prime = 4 * image[:, 0] / (image[:, 0] + 15 * image[:, 1] + 3 * image[:, 2])
        v_prime = 9 * image[:, 1] / (image[:, 0] + 15 * image[:, 1] + 3 * image[:, 2])
        d = 1

    u = 13 * L * (u_prime - .2009)
    v = 13 * L * (v_prime - .4610)

    luv_image = torch.stack((L, u, v), dim=d)

    return luv_image

def luv_to_lch(image):
    # single image
    if len(image.size()) == 3:
        C = torch.pow(torch.pow(image[1],2) + torch.pow(image[2],2), 0.5)
        h = torch.atan2(image[2], image[1])

        lch_image = torch.stack((image[0], C, h))
    # batched image
    if len(image.size()) == 4:
        C = torch.pow(torch.pow(image[:,1],2) + torch.pow(image[:,2],2), 0.5)
        h = torch.atan2(image[:,2], image[:,1])

        lch_image = torch.stack((image[:,0], C, h), dim=1)

    return lch_image

def lch_to_luv(image):

    # single image
    if len(image.size()) == 3:
        u = image[1] * torch.cos(image[2])
        v = image[1] * torch.sin(image[2])

        luv_image = torch.stack((image[0], u, v))
    # batched image
    if len(image.size()) == 4:
        u = image[:,1] * torch.cos(image[:,2])
        v = image[:,1] * torch.sin(image[:,2])

        luv_image = torch.stack((image[:,0], u, v), dim=1)
    return luv_image


def srgb_to_xyz(image):
    return rgb_to_xyz(srgb_to_rgb(image))

def xyz_to_srgb(image):
    return rgb_to_srgb(xyz_to_rgb(image))

def srgb_to_luv(image):
    return xyz_to_luv(srgb_to_xyz(image))

def luv_to_srgb(image):
    return luv_to_xyz(lch_to_luv(image))

def srgb_to_lch(image):
    return luv_to_lch(xyz_to_luv(srgb_to_xyz(image)))

def lch_to_srgb(image):
    return xyz_to_srgb(luv_to_xyz(lch_to_luv(image)))

def sbgr_to_lch(image):
    return srgb_to_lch(BGR_to_RGB(image))

def lch_to_sbgr(image):
    return BGR_to_RGB(lch_to_srgb(image))