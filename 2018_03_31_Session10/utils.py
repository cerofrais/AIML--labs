import copy
import numpy as np

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, "reverse preprocessing"
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im

def alpha_norm(input_matrix, alpha):
    alpha_norm = ((input_matrix.view(-1))**alpha).sum()
    return alpha_norm

def total_variation_norm(input_matrix, beta):
    """
        Total variation norm is the second norm in the paper
        represented as R_V(x)
    """
    to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
    one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
    one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
    total_variation = (((to_check - one_bottom)**2 +
                        (to_check - one_right)**2)**(beta/2)).sum()
    return total_variation

def euclidian_loss(org_matrix, target_matrix):
    distance_matrix = target_matrix - org_matrix
    euclidian_distance = alpha_norm(distance_matrix, 2)
    normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
    return normalized_euclidian_distance

