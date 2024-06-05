from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]


W_ABS_UNARY_KEYS = [
    'no_op',
    "element_wise_sqrt",
    'element_wise_pow',
    'element_wise_exp',
    'element_wise_log',
    'sigmoid',
    'softmax',
]

X_NORM_UNARY_KEYS = [
    'no_op',
    "element_wise_sqrt",
    'element_wise_pow',
    'element_wise_exp',
    'element_wise_log',
    'sigmoid',
    'softmax',   
]

W_ABS_SCALAR_UNARY_KEYS = [
    "no_coe",
    "row_sum", 
    "column_sum", 
    "relative_sum", 
    "to_sum_scalar", 
    "to_mean_scalar",
    "frobenius_norm"
]



# Keep Dimension Matric Unary Operation

def no_op(A: ALLTYPE) -> ALLTYPE:
    return A

def element_wise_sqrt(A: ALLTYPE) -> ALLTYPE:
    return torch.sqrt(A)

def element_wise_pow(A: ALLTYPE) -> ALLTYPE:
    return torch.pow(A, 2)

def element_wise_exp(A: ALLTYPE) -> ALLTYPE:
    return torch.exp(A)

def element_wise_log(A: ALLTYPE) -> ALLTYPE:
    A[A <= 0] == 1
    return torch.log(A)

def sigmoid(A: ALLTYPE) -> ALLTYPE:
    return torch.sigmoid(A)

def softmax(A: ALLTYPE) -> ALLTYPE:
    return F.softmax(A, dim=1)

# Decrease Dimension Unary Operation 

def l1_norm(A: ALLTYPE) -> Vector:
    return torch.norm(A, p=1, dim=0, keepdim=True)

def l2_norm(A: ALLTYPE) -> Vector:
    return torch.norm(A, p=2, dim=0, keepdim=True)

# Coefficient Unary Operation 
def no_coe(A: ALLTYPE) -> Scalar:
    return A / A

def row_sum(A: ALLTYPE) -> Scalar:
    row_sums = torch.sum(A, dim=1)
    expanded_row_sums = row_sums.unsqueeze(1).expand_as(A)
    return 1 / expanded_row_sums

def column_sum(A: ALLTYPE) -> Scalar:
    col_sums = torch.sum(A, dim=0)
    expanded_col_sums = col_sums.unsqueeze(0).expand_as(A)
    return 1 / expanded_col_sums

def relative_sum(A: ALLTYPE) -> Scalar:
    row_sums = torch.sum(A, dim=1)
    col_sums = torch.sum(A, dim=0)
    expanded_row_sums = row_sums.unsqueeze(1).expand_as(A)
    expanded_col_sums = col_sums.unsqueeze(0).expand_as(A)
    return 1 / (expanded_row_sums + expanded_col_sums)
    
def frobenius_norm(A: ALLTYPE) -> Scalar:
    return 1 / torch.norm(A, p='fro')

def to_sum_scalar(A: ALLTYPE) -> Scalar:
    return 1 / torch.sum(A)

def to_mean_scalar(A: ALLTYPE) -> Scalar:
    return 1 / torch.mean(A)


def unary_operation(A, unary_ops):

    unaries = {
        'no_op': no_op,
        "element_wise_sqrt": element_wise_sqrt,
        'element_wise_pow': element_wise_pow,
        'element_wise_exp': element_wise_exp,
        'element_wise_log': element_wise_log,
        'sigmoid': sigmoid,
        'softmax': softmax,  
        "no_coe": no_coe, 
        "row_sum": row_sum, 
        "column_sum": column_sum, 
        "relative_sum": relative_sum, 
        "to_sum_scalar": to_sum_scalar, 
        "to_mean_scalar": to_mean_scalar,
        "frobenius_norm": frobenius_norm,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
    }
    return unaries[unary_ops](A)





