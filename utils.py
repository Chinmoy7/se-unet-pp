# Miscellaneous Utility functions

def dice_coeff(pred, target):
    smooth = 1.
    eps = 1e-10
    num = pred.size(0)
    m1 = pred.float().view(num, -1)  # Flatten
    m2 = target.float().view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2 * intersection) / (m1.sum() + m2.sum()+eps)